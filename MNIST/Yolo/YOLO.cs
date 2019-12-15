using BuildingDetection.LearningCurves;
using CNTK;
using CNTKUtil;
using Newtonsoft.Json;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Text;

namespace BuildingDetection.Yolo
{
    public abstract class Model 
    {
        public Function Network;
        public LearningCurvesData LearningCurvesData;

        public static T LoadModel<T>(string path, ModelFormat format)where T:Model,new()
        {
            var dirPath = Path.GetDirectoryName(path);
            var network = Function.Load(path, DeviceDescriptor.CPUDevice, format);
            var learningCurvesData = LearningCurvesData.LoadFrom(dirPath);
            return new T() {Network=network,LearningCurvesData=learningCurvesData};
        }

        public void SaveModel(string outputPath)
        {
            Directory.CreateDirectory(outputPath);
            Network.Save($"{outputPath}/model.model");
            File.WriteAllText($"{outputPath}/learning.json",JsonConvert.SerializeObject(LearningCurvesData));
        }

        public abstract Learner GetLearner(int epochs,int maxEpochs);
        
        public abstract Function GetLossFunction(Variable truth,Variable prediction);
        
        public abstract Function GetErrorFunction(Variable truth, Variable prediction);

        //public void Train(IData[] training, IData[] testing, Function lossFunc, Function errorFunc, string outputPath, int maxEpochs/*135*/, int batchSize/*64*/, bool autoSave = true, int autoSaveStep = 2)
        //{
        //    // train the model
        //    Console.WriteLine("Epoch\tTrain\tTrain\tTest");
        //    Console.WriteLine("\tLoss\tError\tError");
        //    Console.WriteLine("-----------------------------");

        //    Variable features = Variable.InputVariable(new int[] { (int)H, (int)W, 3 }, DataType.Float, "features");
        //    Variable labels = Variable.InputVariable(new int[] { (int)S, (int)S, (int)B * 5 + (int)C }, DataType.Float, "labels");

        //    var loss = new double[maxEpochs];
        //    var trainingError = new List<double>(maxEpochs);
        //    var testingError = new List<double>(maxEpochs);

        //    for (int epoch = 0; epoch < maxEpochs; epoch++)
        //    {
        //        var learner = GetLearner(epoch, maxEpochs);
        //        var trainer = Network.GetTrainer(learner, lossFunc, errorFunc);
        //        var evaluator = Network.GetEvaluator(errorFunc);
        //        // train one epoch on batches
        //        loss[epoch] = 0.0;
        //        trainingError[epoch] = 0.0;
        //        var batchCount = 0;
        //        training.Index().Shuffle().Batch(batchSize, (indices, begin, end) =>
        //        {
        //            // get the current batch
        //            var featureBatch = features.GetFeaturesBatch(training, indices, begin, end);
        //            var labelBatch = labels.GetLabelsBatch(training, indices, begin, end);

        //            // train the network on the batch
        //            var result = trainer.TrainBatch(
        //                                new[] {
        //                                    (features, featureBatch),
        //                                    (labels,  labelBatch)
        //                                },
        //                                false);
        //            loss[epoch] += result.Loss;
        //            trainingError[epoch] += result.Evaluation;
        //            batchCount++;
        //        });

        //        // show results
        //        loss[epoch] /= batchCount;
        //        trainingError[epoch] /= batchCount;
        //        Console.Write($"{epoch}\t{loss[epoch]:F3}\t{trainingError[epoch]:F3}\t");

        //        // test one epoch on batches
        //        testingError[epoch] = 0.0;
        //        batchCount = 0;
        //        testing.Batch(batchSize, (data, begin, end) =>
        //        {
        //            // get the current batch for testing
        //            var featureBatch = features.GetFeaturesBatch(testing, begin, end);
        //            var labelBatch = labels.GetLabelsBatch(testing, begin, end);

        //            // test the network on the batch
        //            testingError[epoch] += evaluator.TestBatch(
        //                new[] {
        //                    (features, featureBatch),
        //                    (labels,  labelBatch)
        //                }
        //            );
        //            batchCount++;
        //        });
        //        testingError[epoch] /= batchCount;
        //        Console.WriteLine($"{testingError[epoch]:F3}");
        //        if (epoch % autoSaveStep == 0)
        //            SaveModel(outputPath);
        //    }

        //    // show final results
        //    var finalError = testingError.Last();
        //    Console.WriteLine();
        //    Console.WriteLine($"Final test error: {finalError:0.00}");
        //    Console.WriteLine($"Final test accuracy: {1 - finalError:0.00}");
        //    SaveModel(outputPath);
        //}

        public void TrainMinibatchSource(IData[] training, IData[] testing, string outputPath, int maxEpochs/*135*/, int batchSize/*64*/, bool autoSave = true, int autoSaveStep = 2)
        {
            // train the model
            Console.WriteLine("Epoch\tTrain\tTrain\tTest");
            Console.WriteLine("\tLoss\tError\tError");
            Console.WriteLine("-----------------------------");

            var loss = Enumerable.Repeat(0d, maxEpochs).ToList();
            var trainingError = Enumerable.Repeat(0d, maxEpochs).ToList();
            var testingError = Enumerable.Repeat(0d, maxEpochs).ToList();
           
            Variable features= Network.Arguments[0];
            Variable labels = Variable.InputVariable(Network.Output.Shape, Network.Output.DataType,"labels");
            
            Function lossFunc = GetLossFunction(labels,Network.Output);
            Function errorFunc = GetErrorFunction(labels,Network.Output);
            
            var (imageMap_train, dataMap_train) = CreateMapFiles(training);
            var (imageMap_test, dataMap_test) = CreateMapFiles(testing);

            for (int epoch = 0; epoch < maxEpochs; epoch++)
            {
                var learner = GetLearner(epoch, maxEpochs);
                var trainer = Network.GetTrainer(learner, lossFunc, errorFunc);

                var source = CreateMinibatchSource(imageMap_train, dataMap_train);
                var featureStreamInfo = source.StreamInfo("image");
                var labelStreamInfo = source.StreamInfo("boundingbox");
                int batchCant = 0;
                while (trainer.TotalNumberOfSamplesSeen() < training.Length) 
                {
                    var batch = source.GetNextMinibatch((uint)batchSize, DeviceDescriptor.GPUDevice(0));
                    // train the network on the batch
                    var result = trainer.TrainMinibatch(new Dictionary<Variable, MinibatchData>() {
                                                                { features, batch[featureStreamInfo]},
                                                                { labels, batch[labelStreamInfo]}},
                                                                DeviceDescriptor.GPUDevice(0));

                    loss[epoch] += trainer.PreviousMinibatchLossAverage();
                    trainingError[epoch] += trainer.PreviousMinibatchEvaluationAverage();
                    batchCant++;
                }
                // show results
                loss[epoch] /= batchCant;
                trainingError[epoch] /= batchCant;
                Console.Write($"{epoch}\t{loss[epoch]:F3}\t{trainingError[epoch]:F3}\t");

                var evaluator = Network.GetEvaluator(errorFunc);
                source = CreateMinibatchSource(imageMap_test, dataMap_test);
                featureStreamInfo = source.StreamInfo("image");
                labelStreamInfo = source.StreamInfo("boundingbox");
                batchCant = 10;
                for (int batchCount = 0; batchCount < batchCant; batchCount++)
                {
                    var batch = source.GetNextMinibatch((uint)batchSize, DeviceDescriptor.CPUDevice);
                    // test the network on the batch
                    testingError[epoch] += evaluator.TestMinibatch(new UnorderedMapVariableMinibatchData() {
                                                                    { features, batch[featureStreamInfo]},
                                                                    { labels, batch[labelStreamInfo]}},
                                                                    DeviceDescriptor.GPUDevice(0));
                }
                testingError[epoch] /= batchCant;
                Console.WriteLine($"{testingError[epoch]:F3}");
                if (epoch % autoSaveStep == 0)
                    SaveModel(outputPath);
            }

            // show final results
            var finalError = testingError.Last();
            Console.WriteLine();
            Console.WriteLine($"Final test error: {finalError:0.00}");
            Console.WriteLine($"Final test accuracy: {1 - finalError:0.00}");
            SaveModel(outputPath);
        }

        private static (string imagemap, string labelmap) CreateMapFiles(IData[] data)
        {
            string imagemap = "imagesmap.txt";
            string labelmap = "labelsmap.txt";
            StringBuilder images = new StringBuilder();
            StringBuilder labels = new StringBuilder();
            foreach (var item in data)
            {
                images.AppendLine(item.Reference + "\t 0");
                labels.AppendLine("|boundingbox " + string.Join(' ', item.Labels));
            }
            File.WriteAllText(imagemap, images.ToString());
            File.WriteAllText(labelmap, labels.ToString());
            return (imagemap, labelmap);
        }

        protected abstract MinibatchSource CreateMinibatchSource(params string[] map_file);

        public abstract void Evaluate(IData item);

    }

    public class YOLO:Model
    {
        public static readonly string[] Tags=new [] {"building"};
        public static Color[] TagColors=new[] {Color.Red};
        public static readonly int S =7;
        public static readonly int B =2;
        public static readonly int C =1;
        public static readonly int H =448;
        public static readonly int W=448;
        
        public YOLO()
        {
            //buildNetwork alo LINQ
            Variable features = Variable.InputVariable(new int[] { (int)H, (int)W, 3 }, DataType.Float, "features");
            Func<Variable, Function> leakyRelu = (Variable v) => CNTKLib.LeakyReLU(v, 0.1);

            Network=features.Convolution2D(64, new[] { 7, 7 }, strides: new[] { 2 }, activation: leakyRelu)
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
                            .Dense(new[] { (int)S, (int)S, (int)B * 5 + (int)C })
                            .ToNetwork();
        }

        public override Learner GetLearner(int epoch,int maxEpochs) 
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
                Function classesPrediction = CNTKLib.Slice(prediction, new AxisVector() { new Axis(2) }, new IntVector() { 5*B }, new IntVector() { 5*B+C });
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
                int anchorOffset = 5  * i;
                Function objectnessTruth = CNTKLib.Slice(truth, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 4 }, new IntVector() { anchorOffset + 5 });
                Function objectnessPrediction = CNTKLib.Slice(prediction, new AxisVector() { new Axis(2) }, new IntVector() { anchorOffset + 4 }, new IntVector() { anchorOffset + 5 });
                objectnessPrediction = CNTKLib.GreaterEqual(objectnessPrediction, thresholdChannel); // convert to 0 where the value is below threshold, and 1 if it is above threshold

                Function noObjectnessTruth = CNTKLib.Minus(onesChannel, objectnessTruth);
                Function noObjectnessPrediction = CNTKLib.Minus(onesChannel, objectnessPrediction);

                // todo class prediction should be set to false when it is below threshold
                Function classesTruth = CNTKLib.Slice(truth, new AxisVector() { new Axis(2) }, new IntVector() { 5 * B }, new IntVector() { 5*B + C});
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

        private double GetLearningRate(int epoch,int maxEpochs)
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
