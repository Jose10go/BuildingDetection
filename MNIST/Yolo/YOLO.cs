using CNTK;
using CNTKUtil;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace BuildingDetection.Yolo
{
    public static class YOLO
    {
        public static string[] Tags = new string[] { "building" };
        public static Color[] TagColors = new Color[] {Color.Red};
        public static int S = 7;
        public static int B = 2;
        public static int C = Tags.Length;
        public static int H = 448;
        public static int W = 448;
        public static Variable features= Variable.InputVariable(new int[] { H, W, 3 }, DataType.Float, "features");
        public static Variable labels = Variable.InputVariable(new int[] { S, S, B * 5 + C }, DataType.Float, "labels");
        public static Function BuildYoloDNN()
        {
            //buildNetwork alo LINQ
            Func<Variable, Function> leakyRelu = (Variable v) => CNTKLib.LeakyReLU(v, 0.1);

            var network = features.Convolution2D(64, new[] { 7, 7 }, strides: new[] { 2 }, activation: leakyRelu)
                                  .Pooling(PoolingType.Max, new[] { 2, 2 }, new[] { 2 })
                                  .Convolution2D(192, new[] { 3, 3 }, activation: leakyRelu)
                                  .Pooling(PoolingType.Max, new[] { 2, 2 }, new[] { 2 })
                                  .Convolution2D(128, new[] { 1, 1 }, activation: leakyRelu)
                                  .Convolution2D(256, new[] { 3, 3 }, activation: leakyRelu)
                                  .Convolution2D(256, new[] { 1, 1 }, activation: leakyRelu)
                                  .Convolution2D(512, new[] { 3, 3 }, activation: leakyRelu)
                                  .Pooling(PoolingType.Max, new[] { 2, 2 }, new[] { 2 })
                                  .Convolution2D(256, new[] { 1, 1 }, activation: leakyRelu)
                                  .Convolution2D(512, new[] { 3, 3 }, activation: leakyRelu)
                                  .Convolution2D(256, new[] { 1, 1 }, activation: leakyRelu)
                                  .Convolution2D(512, new[] { 3, 3 }, activation: leakyRelu)
                                  .Convolution2D(256, new[] { 1, 1 }, activation: leakyRelu)
                                  .Convolution2D(512, new[] { 3, 3 }, activation: leakyRelu)
                                  .Convolution2D(256, new[] { 1, 1 }, activation: leakyRelu)
                                  .Convolution2D(512, new[] { 3, 3 }, activation: leakyRelu)
                                  .Convolution2D(512, new[] { 1, 1 }, activation: leakyRelu)
                                  .Convolution2D(1024, new[] { 3, 3 }, activation: leakyRelu)
                                  .Pooling(PoolingType.Max, new[] { 2, 2 }, new[] { 2 })
                                  .Convolution2D(512, new[] { 1, 1 }, activation: leakyRelu)
                                  .Convolution2D(1024, new[] { 3, 3 }, activation: leakyRelu)
                                  .Convolution2D(512, new[] { 1, 1 }, activation: leakyRelu)
                                  .Convolution2D(1024, new[] { 3, 3 }, activation: leakyRelu)
                                  .Convolution2D(1024, new[] { 3, 3 }, activation: leakyRelu)
                                  .Convolution2D(1024, new[] { 3, 3 }, strides: new[] { 2 }, activation: leakyRelu)
                                  .Convolution2D(1024, new[] { 3, 3 }, activation: leakyRelu)
                                  .Convolution2D(1024, new[] { 3, 3 }, activation: leakyRelu)
                                  .Dense(new[] { 4096 }, activation: leakyRelu)
                                  .Dense(new[] { S, S, B * 5 + C })
                                  .ToNetwork();

            return network;
        }
       
        public static Learner GetYoloLearner(this Function network,int epoch,int maxEpochs) 
        {
            return CNTKLib.MomentumSGDLearner(
                        new ParameterVector((ICollection)network.Parameters()),
                        new TrainingParameterScheduleDouble(GetLearningRate(epoch, maxEpochs)),
                        new TrainingParameterScheduleDouble(0.9));
        }

        public static Function GetYoloLossFunction(this Function network) 
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
                Function objectnessTruth = CNTKLib.Slice(labels, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 4 }, new IntVector() { anchorOffset + 5 });
                NDShape singleChannelShape = objectnessTruth.Output.Shape.SubShape(0, 3);

                // noobjectnessTruth is the "1^(noobj)" from loss function in the paper
                Parameter onesChannel = new Parameter(singleChannelShape, DataType.Float, CNTKLib.ConstantInitializer(1f), DeviceDescriptor.CPUDevice);
                Function noobjectnessTruth = CNTKLib.Minus(onesChannel, objectnessTruth);

                // coordinatesLambdaChannel is the LAMBDA(coord) from the loss function in the paper
                Parameter coordinatesLambdaChannel = new Parameter(singleChannelShape, DataType.Float, CNTKLib.ConstantInitializer(coordinatesLambda), DeviceDescriptor.CPUDevice);

                // loss due to differences in centers
                Function centerPrediction = CNTKLib.Slice(network.Output, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset }, new IntVector() { anchorOffset + 2 });
                Function centerTruth = CNTKLib.Slice(labels, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset }, new IntVector() { anchorOffset + 2 });
                Function centerDifference = CNTKLib.Minus(centerPrediction, centerTruth);
                Function centerLoss = CNTKLib.ElementTimes(coordinatesLambdaChannel, CNTKLib.ElementTimes(objectnessTruth, CNTKLib.Square(centerDifference)));
                deltaChannels.Add(centerLoss); // Add 1 Function of 2 channels to the vector

                // loss due to differences in width/height
                Function dimensionPrediction = CNTKLib.Slice(network.Output, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 2 }, new IntVector() { anchorOffset + 4 });
                Function dimensionTruth = CNTKLib.Slice(labels, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 2 }, new IntVector() { anchorOffset + 4 });
                Function dimensionSqrtDifference = CNTKLib.Minus(CNTKLib.Sqrt(dimensionPrediction), CNTKLib.Sqrt(dimensionTruth));
                Function dimensionLoss = CNTKLib.ElementTimes(coordinatesLambdaChannel, CNTKLib.ElementTimes(objectnessTruth, CNTKLib.Square(dimensionSqrtDifference)));
                deltaChannels.Add(dimensionLoss); // Add 1 Function of 2 channels to the vector

                // noObjectLambdaChannel is the LAMBDA(noobj) from the loss function in the paper
                Parameter noObjectLambdaChannel = new Parameter(singleChannelShape, DataType.Float, CNTKLib.ConstantInitializer(noObjectLambda), DeviceDescriptor.CPUDevice);
                // objectnessWeight is the combination of 1^(obj) + LAMBDA(noobj) * 1^(noobj)
                Function objectnessWeight = CNTKLib.Plus(objectnessTruth, CNTKLib.ElementTimes(noObjectLambdaChannel, noobjectnessTruth)); // todo this could potentially be wrong as the documentation describes that this is an element wise BINARY addition

                // loss due to difference in objectness
                Function objectnessPrediction = CNTKLib.Slice(network.Output, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 4 }, new IntVector() { anchorOffset + 5 });
                Function objectnessDifference = CNTKLib.Minus(objectnessPrediction, objectnessTruth);
                Function objectnessLoss = CNTKLib.ElementTimes(objectnessWeight, CNTKLib.Square(objectnessDifference));
                deltaChannels.Add(objectnessLoss); // Add 1 Function of 1 channels to the vector

                // loss due to difference in class predictions
                Function classesPrediction = CNTKLib.Slice(network.Output, new AxisVector() { new Axis(2) }, new IntVector() { 5*B }, new IntVector() { 5*B+C });
                Function classesTruth = CNTKLib.Slice(labels, new AxisVector() { new Axis(2) }, new IntVector() { 5 * B }, new IntVector() { 5 * B + C });
                Function classesDifference = CNTKLib.Minus(classesPrediction, classesTruth);
                Function classesLoss = CNTKLib.ElementTimes(objectnessTruth, CNTKLib.Square(classesDifference));
                deltaChannels.Add(classesLoss); // Add 1 Function of "classes" channels to the vector
            }

            // As described above the delta function contains all losses
            Function delta = CNTKLib.Splice(deltaChannels, new Axis(2));

            ////the loss Function can't be a multidimensional tensor, then reduce the dimension by taking the sum
            Function totalSum = CNTKLib.ReduceSum(delta, new AxisVector() { new Axis(2), new Axis(1), new Axis(0) });
            return CNTKLib.Reshape(totalSum, new int[] { 1 });

        }

        public static Function GetYoloErrorFunction(this Function network,float threshold=0.3f)
        {
            var prediction = network.Output;
            // thresholdChannel is a one channel parameter filled with the threshold value
            NDShape singleChannelShape = new int[] { prediction.Shape[0], prediction.Shape[1], 1 };
            Parameter thresholdChannel = new Parameter(singleChannelShape, DataType.Float, CNTKLib.ConstantInitializer(threshold), DeviceDescriptor.CPUDevice);
            Parameter onesChannel = new Parameter(singleChannelShape, DataType.Float, CNTKLib.ConstantInitializer(1f), DeviceDescriptor.CPUDevice);

            VariableVector truePredictionChannels = new VariableVector();
            VariableVector falsePredictionChannels = new VariableVector();
            for (int i = 0; i < B; i++)
            {
                int anchorOffset = 5  * i;

                Function objectnessTruth = CNTKLib.Slice(labels, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 4 }, new IntVector() { anchorOffset + 5 });
                Function objectnessPrediction = CNTKLib.Slice(prediction, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 4 }, new IntVector() { anchorOffset + 5 });
                objectnessPrediction = CNTKLib.GreaterEqual(objectnessPrediction, thresholdChannel); // convert to 0 where the value is below threshold, and 1 if it is above threshold

                Function noObjectnessTruth = CNTKLib.Minus(onesChannel, objectnessTruth);
                Function noObjectnessPrediction = CNTKLib.Minus(onesChannel, objectnessPrediction);

                // todo class prediction should be set to false when it is below threshold
                Function classesTruth = CNTKLib.Slice(labels, new AxisVector() { new Axis(2) }, new IntVector() { 5 * B }, new IntVector() { 5*B + C});
                classesTruth = CNTKLib.Argmax(classesTruth, new Axis(2)); // Reduce the tensor to one channel; filled with the index of the maximum value (= the index of the class)
                Function classesPrediction = CNTKLib.Slice(prediction, new AxisVector() { new Axis(2) }, new IntVector() { 5*B }, new IntVector() { 5*B + C});
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

        private static double GetLearningRate(int epoch,int maxEpochs)
        {
            if (epoch < 55 * maxEpochs / 100)
                return 1e-2;
            if (epoch < 77 * maxEpochs / 100)
                return 1e-3;
            return 1e-4;
        }

        //private static Function YoloLosssFunc(Variable yolo_output,Constant true_boxes, Constant detectors_mask,Constant matching_true_boxes,int[] anchors,int num_classes)
        //{
        //    var num_anchors = anchors.Length;
        //    var object_scale = 5;
        //    var no_object_scale = 1;
        //    var class_scale = 1;
        //    var coordinates_scale = 1;

        //    var (pred_xy, pred_wh, pred_confidence, pred_class_prob) = yolo_head(output, anchors, num_classes);

        //    var yolo_output_shape = yolo_output.Shape;
        //    var feats = yolo_output.Reshape(new NDShape(new SizeTVector(new[] { -1, yolo_output_shape[1], yolo_output_shape[2], num_anchors, num_classes + 5 })));
        //    var pred_boxes = CNTKLib(K.sigmoid(CNTKLib.Slice(feats,feats,0,2), feats[feats,2,4]), axis = -1);
        //}

        //private static object yolo_head(Variable yolo_output, int[] anchors, int num_classes)
        //{
        //    throw new NotImplementedException();
        //}

        //private static (object yolo_output, object true_boxes, object detectors_mask, object matching_true_boxes) OutputSplit(Variable output)
        //{
        //    throw new NotImplementedException();
        //}

    }
}
