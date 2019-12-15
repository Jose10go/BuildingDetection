using BuildingDetection.LearningCurves;
using BuildingDetection.Yolo;
using CNTK;
using CNTKUtil;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;

namespace BuildingDetection.TinyYoloV2
{
    public class TinyYOLOV2 : Model
    {
        public static readonly string[] Tags = new[] { "building" };
        public static Color[] TagColors = new[] { Color.Red };
        public static readonly int S = 7;
        public static readonly int B = 2;
        public static readonly int C = 1;
        public static readonly int H = 448;
        public static readonly int W = 448;

        public TinyYOLOV2(Function network, LearningCurvesData learningCurvesData)
        {
            Network = network;
            LearningCurvesData = learningCurvesData;
        }

        public static TinyYOLOV2 CreateTransferLearningModel<T>(string baseModelFile, string hiddenNodeName, ParameterCloningMethod parameterCloningMethod = ParameterCloningMethod.Freeze) where T : Model, new()
        {
            Function baseModel = Function.Load(baseModelFile, DeviceDescriptor.CPUDevice, ModelFormat.ONNX);
            Function lastNode = baseModel.FindByName(hiddenNodeName);

            // Clone the desired layers with fixed weights
            Variable network = CNTKLib.Combine(new VariableVector() { lastNode }).Clone(parameterCloningMethod);
            Variable transpose = network.TransposeAxes(new Axis(3), new Axis(2))
                                        .TransposeAxes(new Axis(2), new Axis(1))
                                        .TransposeAxes(new Axis(1), new Axis(0));
            var n = transpose.Convolution(new[] { 1, 1, 1, 1024, B * (5 + C) }, true, false, new int[] { 1, 1, 1, 1024 });
            Console.WriteLine(n.ToSummary());

            n.Save("tl101.model");
            return new TinyYOLOV2(network, new LearningCurvesData());
        }

        public override Learner GetLearner(int epoch, int maxEpochs)
        {
            return CNTKLib.MomentumSGDLearner(
                        new ParameterVector((ICollection)Network.Parameters()),
                        new TrainingParameterScheduleDouble(GetLearningRate(epoch, maxEpochs)),
                        new TrainingParameterScheduleDouble(0.9));
        }

        public override Function GetLossFunction(Variable truth, Variable prediction)
        {
            float coordinatesLambda = 5f;
            float noObjectLambda = .5f;

            // we create a new function with the same dimension as the prediction variable. 
            // This new "delta" function (see after the for loop) contains all errors which are calculated by the Yolo v2 loss function. See here: https://pjreddie.com/media/files/papers/yolo_1.pdf
            // For example: 
            //  When a particular cell value in the prediction Varibale contain the dx coordinate, 
            //  then this cell in the "delta" function will contain the squared x-error (when there was an object detection for that anchor box)

            VariableVector deltaChannels = new VariableVector();
            for (int i = 0; i < B; i++)
            {
                int anchorOffset = 5 * i;

                // objectnessTruth is the "1^(obj)" from loss function in the paper
                Function objectnessTruth = CNTKLib.Slice(truth, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 4 }, new IntVector() { anchorOffset + 5 });
                NDShape singleChannelShape = objectnessTruth.Output.Shape.SubShape(0, 3);

                // noobjectnessTruth is the "1^(noobj)" from loss function in the paper
                Parameter onesChannel = new Parameter(singleChannelShape, DataType.Float, CNTKLib.ConstantInitializer(1f), DeviceDescriptor.CPUDevice);
                Function noobjectnessTruth = CNTKLib.Minus(onesChannel, objectnessTruth);

                // coordinatesLambdaChannel is the LAMBDA(coord) from the loss function in the paper
                Parameter coordinatesLambdaChannel = new Parameter(singleChannelShape, DataType.Float, CNTKLib.ConstantInitializer(coordinatesLambda), DeviceDescriptor.CPUDevice);

                // loss due to differences in centers
                Function centerPrediction = CNTKLib.Slice(prediction, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset }, new IntVector() { anchorOffset + 2 });
                Function centerTruth = CNTKLib.Slice(truth, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset }, new IntVector() { anchorOffset + 2 });
                Function centerDifference = CNTKLib.Minus(centerPrediction, centerTruth);
                Function centerLoss = CNTKLib.ElementTimes(coordinatesLambdaChannel, CNTKLib.ElementTimes(objectnessTruth, CNTKLib.Square(centerDifference)));
                deltaChannels.Add(centerLoss); // Add 1 Function of 2 channels to the vector

                // loss due to differences in width/height
                Function dimensionPrediction = CNTKLib.Slice(prediction, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 2 }, new IntVector() { anchorOffset + 4 });
                Function dimensionTruth = CNTKLib.Slice(truth, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 2 }, new IntVector() { anchorOffset + 4 });
                Function dimensionSqrtDifference = CNTKLib.Minus(CNTKLib.Sqrt(dimensionPrediction), CNTKLib.Sqrt(dimensionTruth));
                Function dimensionLoss = CNTKLib.ElementTimes(coordinatesLambdaChannel, CNTKLib.ElementTimes(objectnessTruth, CNTKLib.Square(dimensionSqrtDifference)));
                deltaChannels.Add(dimensionLoss); // Add 1 Function of 2 channels to the vector

                // noObjectLambdaChannel is the LAMBDA(noobj) from the loss function in the paper
                Parameter noObjectLambdaChannel = new Parameter(singleChannelShape, DataType.Float, CNTKLib.ConstantInitializer(noObjectLambda), DeviceDescriptor.CPUDevice);
                // objectnessWeight is the combination of 1^(obj) + LAMBDA(noobj) * 1^(noobj)
                Function objectnessWeight = CNTKLib.Plus(objectnessTruth, CNTKLib.ElementTimes(noObjectLambdaChannel, noobjectnessTruth));

                // loss due to difference in objectness
                Function objectnessPrediction = CNTKLib.Slice(prediction, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 4 }, new IntVector() { anchorOffset + 5 });
                Function objectnessDifference = CNTKLib.Minus(objectnessPrediction, objectnessTruth);
                Function objectnessLoss = CNTKLib.ElementTimes(objectnessWeight, CNTKLib.Square(objectnessDifference));
                deltaChannels.Add(objectnessLoss); // Add 1 Function of 1 channels to the vector

                // loss due to difference in class predictions
                Function classesPrediction = CNTKLib.Slice(prediction, new AxisVector() { new Axis(2) }, new IntVector() { 5 * B }, new IntVector() { 5 * B + C });
                Function classesTruth = CNTKLib.Slice(truth, new AxisVector() { new Axis(2) }, new IntVector() { 5 * B }, new IntVector() { 5 * B + C });
                Function classesDifference = CNTKLib.Minus(classesPrediction, classesTruth);
                Function classesLoss = CNTKLib.ElementTimes(objectnessTruth, CNTKLib.Square(classesDifference));
                deltaChannels.Add(classesLoss); // Add 1 Function of "classes" channels to the vector
            }

            // As described above the delta function contains all losses
            Function delta = CNTKLib.Splice(deltaChannels, new Axis(2));
            return delta;
            ////if the loss Function can't be a multidimensional tensor, then reduce the dimension by taking the sum
            //Function totalSum = CNTKLib.ReduceSum(delta, new AxisVector() { new Axis(2), new Axis(1), new Axis(0) });
            //return CNTKLib.Reshape(totalSum, new int[] { 1 });

        }

        public override Function GetErrorFunction(Variable truth, Variable prediction)
        {
            float threshold = 0.3f;
            // thresholdChannel is a one channel parameter filled with the threshold value
            NDShape singleChannelShape = new int[] { prediction.Shape[0], prediction.Shape[1], 1 };
            Parameter thresholdChannel = new Parameter(singleChannelShape, DataType.Float, CNTKLib.ConstantInitializer(threshold), DeviceDescriptor.CPUDevice);
            Parameter onesChannel = new Parameter(singleChannelShape, DataType.Float, CNTKLib.ConstantInitializer(1f), DeviceDescriptor.CPUDevice);

            VariableVector truePredictionChannels = new VariableVector();
            VariableVector falsePredictionChannels = new VariableVector();
            for (int i = 0; i < B; i++)
            {
                int anchorOffset = 5 * i;
                Function objectnessTruth = CNTKLib.Slice(truth, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 4 }, new IntVector() { anchorOffset + 5 });
                Function objectnessPrediction = CNTKLib.Slice(prediction, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 4 }, new IntVector() { anchorOffset + 5 });
                objectnessPrediction = CNTKLib.GreaterEqual(objectnessPrediction, thresholdChannel); // convert to 0 where the value is below threshold, and 1 if it is above threshold

                Function noObjectnessTruth = CNTKLib.Minus(onesChannel, objectnessTruth);
                Function noObjectnessPrediction = CNTKLib.Minus(onesChannel, objectnessPrediction);

                // todo class prediction should be set to false when it is below threshold
                Function classesTruth = CNTKLib.Slice(truth, new AxisVector() { new Axis(2) }, new IntVector() { 5 * B }, new IntVector() { 5 * B + C });
                classesTruth = CNTKLib.Argmax(classesTruth, new Axis(2)); // Reduce the tensor to one channel; filled with the index of the maximum value (= the index of the class)
                Function classesPrediction = CNTKLib.Slice(prediction, new AxisVector() { new Axis(2) }, new IntVector() { 5 * B }, new IntVector() { 5 * B + C });
                classesPrediction = CNTKLib.Argmax(classesPrediction, new Axis(2)); // Reduce the tensor to one channel; filled with the index of the maximum value (= the index of the class)

                // True Positives (tensor contains 1 if yes, 0 otherwise) 
                Function truePositives = CNTKLib.ElementTimes(objectnessTruth, objectnessPrediction); // should detect an object, and detects an object
                truePositives = CNTKLib.ElementTimes(truePositives, CNTKLib.Equal(classesTruth, classesPrediction)); // and should detecect the same class

                // True Negatives (tensor contains 1 if yes, 0 otherwise)
                Function trueNegatives = CNTKLib.ElementTimes(noObjectnessTruth, noObjectnessPrediction); // should detect no object, and detects no object

                // False Negatives (tensor contains 1 if yes, 0 otherwise)
                Function falseNegatives = CNTKLib.ElementTimes(objectnessTruth, noObjectnessPrediction); // should detect an object, but detects no object

                // False Positives (tensor contains 1 if yes, 0 otherwise)
                Function falsePositives = CNTKLib.ElementTimes(noObjectnessTruth, objectnessPrediction); // should detect no object, but detects an object

                // Missclassification (tensor contains 1 if yes, 0 otherwise)
                Function missclassifications = CNTKLib.ElementTimes(objectnessTruth, objectnessPrediction); // should detect an object, and detects an object
                missclassifications = CNTKLib.ElementTimes(missclassifications, CNTKLib.NotEqual(classesTruth, classesPrediction)); // but detects a different class

                truePredictionChannels.Add(truePositives);
                truePredictionChannels.Add(trueNegatives);

                falsePredictionChannels.Add(falseNegatives);
                falsePredictionChannels.Add(falsePositives);
                falsePredictionChannels.Add(missclassifications);
            }

            Function allGoodPredictions = CNTKLib.Splice(truePredictionChannels, new Axis(2));
            Function allbadPredictions = CNTKLib.Splice(falsePredictionChannels, new Axis(2));

            // sum the variable vector
            Function goodPredictionSum = CNTKLib.ReduceSum(allGoodPredictions, new AxisVector() { new Axis(2), new Axis(1), new Axis(0) });
            Function badPredictionSum = CNTKLib.ReduceSum(allbadPredictions, new AxisVector() { new Axis(2), new Axis(1), new Axis(0) });

            // fail safe: The value of goodPredictionSum isn't allowed to be 0, because the division at the end would fail.
            Parameter one = new Parameter(new int[] { 1, 1, 1 }, DataType.Float, CNTKLib.ConstantInitializer(1f), DeviceDescriptor.CPUDevice);
            goodPredictionSum = CNTKLib.ElementMax(goodPredictionSum, one, "Fail safe");

            // device the summations (Take into account the case where there are 0 true predictions
            return CNTKLib.ElementDivide(badPredictionSum, goodPredictionSum);
        }

        private double GetLearningRate(int epoch, int maxEpochs)
        {
            if (epoch < 55 * maxEpochs / 100)
                return 1e-2;
            if (epoch < 77 * maxEpochs / 100)
                return 1e-3;
            return 1e-4;
        }

        protected override MinibatchSource CreateMinibatchSource(params string[] map_file)
        {
            string image_map_file = map_file[0];
            string label_map_file = map_file[1];
            List<CNTKDictionary> transforms = new List<CNTKDictionary>
            {
                CNTKLib.ReaderScale((int)W,(int)H,3, "linear")
            };
            var ctfDeserializer = CNTKLib.CTFDeserializer(label_map_file, new StreamConfigurationVector() { new StreamConfiguration("boundingbox", S * S * (5 * B + C)) });
            var imageDeserializer = CNTKLib.ImageDeserializer(image_map_file, "label", 1, "image", transforms);
            MinibatchSourceConfig config = new MinibatchSourceConfig(new List<CNTKDictionary> { imageDeserializer, ctfDeserializer });

            return CNTKLib.CreateCompositeMinibatchSource(config);
        }

        public override void Evaluate(IData item)
        {
            var inputs = new Dictionary<Variable, Value>();
            inputs.Add(Network.Arguments.First(), new Value(new NDArrayView(new NDShape(new SizeTVector() { (uint)W, (uint)H, 3 }), item.Features, DeviceDescriptor.CPUDevice)));
            var outputs = new Dictionary<Variable, Value>() { { Network.Output, null } };
            Network.Evaluate(inputs, outputs, DeviceDescriptor.CPUDevice);
            var boundingboxes = YOLOOutputParser.ParseOutputs(outputs[Network.Output].GetDenseData<float>(Network.Output)[0].ToArray());
            boundingboxes = YOLOOutputParser.FilterBoundingBoxes(boundingboxes, 3, 0.1f);
            var dir = Path.GetDirectoryName(item.Reference);
            var imgname = Path.GetFileName(item.Reference);
            YoloBoundingBox.DrawBoundingBox(dir, Path.Combine(dir, "out"), imgname, boundingboxes);
        }

    }
}
