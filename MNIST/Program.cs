using BuildingDetection.LearningCurves;
using BuildingDetection.Utils;
using BuildingDetection.Yolo;
using CNTK;
using CNTKUtil;
using MNIST.Utils;
using OxyPlot;
using OxyPlot.Series;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static BuildingDetection.Yolo.YOLO;

namespace MNIST
{
    class Program
    {

        static void Main(string[] args)
        {
            //BuildAndTrainYOLO();
            //TestDrawer();
            //TestDrawLearningCurve();
            //TestDrawLearningCurve2();
        }
        private static void TestDrawLearningCurve2()
        {
            LearningCurves.ExportSvgFromData("learning.json");
        }

        private static void TestDrawLearningCurve()
        {
            List<double> training_error = new List<double> { 0.14, 0.14, 0.15, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16 };
            List<double> testing_error = new List<double> { 0.19, 0.19, 0.20, 0.20, 0.21, 0.21, 0.21, 0.21, 0.21, 0.21 };

            var learn = new LearningCurves(training_error, testing_error);
            learn.ExportSVG();
        }

        static void TestDrawer()
        {
            var (training, _ ) = LoadData();
            foreach (AnnotationImage item in training)
            {
                var path = Path.GetDirectoryName(item.FileName);
                var outpath = Path.Combine(path, "out");
                var file = Path.GetFileName(item.FileName);
                YoloBoundingBox.DrawBoundingBox(path, outpath, file, item.Object);
            }
        }

        static void BuildAndTrainYOLO() 
        {
            var (training, testing) = LoadData();

            //build network
            var network = BuildYoloDNN();
            Console.WriteLine("Model architecture:");
            Console.WriteLine(network.ToSummary());

            // set up the loss function and the error function
            var lossFunc = network.GetYoloLossFunction();
            var errorFunc = network.GetYoloErrorFunction();
            Train(network, training, testing, lossFunc, errorFunc, "yolo", 10, 10);
        }

        static (IData[] training,IData[] testing) LoadData(double percent = 0.8) 
        {
            //// load training and testing data
            Console.WriteLine("Loading data....");
            var l = ReadAnnotationImages.ReadFromDirectory(@"C:\Users\Jose10go\Downloads\dataset", 416, 416);
            var split = (int)(percent * l.Count);
            var training = l.Take(split).ToArray();
            var testing = l.Skip(split).ToArray();
            return (training, testing);
        }

        static void Train(Function network,IData[] training,IData[] testing,Function lossFunc,Function errorFunc,string outputPath,int maxEpochs/*135*/, int batchSize/*64*/,bool autoSave=true,int autoSaveStep=2) 
        {
            // train the model
            Console.WriteLine("Epoch\tTrain\tTrain\tTest");
            Console.WriteLine("\tLoss\tError\tError");
            Console.WriteLine("-----------------------------");

            var loss = new double[maxEpochs];
            var trainingError = new List<double>(maxEpochs);
            var testingError = new List<double>(maxEpochs);

            for (int epoch = 0; epoch < maxEpochs; epoch++)
            {
                var learner = network.GetYoloLearner(epoch,maxEpochs);
                var trainer = network.GetTrainer(learner, lossFunc, errorFunc);
                var evaluator = network.GetEvaluator(errorFunc);
                // train one epoch on batches
                loss[epoch] = 0.0;
                trainingError[epoch] = 0.0;
                var batchCount = 0;
                training.Index().Shuffle().Batch(batchSize, (indices, begin, end) =>
                {
                    // get the current batch
                    var featureBatch = features.GetFeaturesBatch(training, indices, begin, end);
                    var labelBatch = labels.GetLabelsBatch(training, indices, begin, end);

                    // train the network on the batch
                    var result = trainer.TrainBatch(
                                        new[] {
                                            (features, featureBatch),
                                            (labels,  labelBatch)
                                        },
                                        false);
                    loss[epoch] += result.Loss;
                    trainingError[epoch] += result.Evaluation;
                    batchCount++;
                });

                // show results
                loss[epoch] /= batchCount;
                trainingError[epoch] /= batchCount;
                Console.Write($"{epoch}\t{loss[epoch]:F3}\t{trainingError[epoch]:F3}\t");

                // test one epoch on batches
                testingError[epoch] = 0.0;
                batchCount = 0;
                testing.Batch(batchSize, (data, begin, end) =>
                {
                    // get the current batch for testing
                    var featureBatch = features.GetFeaturesBatch(testing, begin, end);
                    var labelBatch = labels.GetLabelsBatch(testing, begin, end);

                    // test the network on the batch
                    testingError[epoch] += evaluator.TestBatch(
                        new[] {
                            (features, featureBatch),
                            (labels,  labelBatch)
                        }
                    );
                    batchCount++;
                });
                testingError[epoch] /= batchCount;
                Console.WriteLine($"{testingError[epoch]:F3}");
                if (epoch % autoSaveStep==0)
                    SaveModel(network,trainingError,testingError, outputPath);
            }

            // show final results
            var finalError = testingError.Last();
            Console.WriteLine();
            Console.WriteLine($"Final test error: {finalError:0.00}");
            Console.WriteLine($"Final test accuracy: {1 - finalError:0.00}");
            SaveModel(network,trainingError,testingError, outputPath);
        }

        private static void SaveModel(Function network, List<double> trainingError, List<double> testingError, string outputPath)
        {
            Directory.CreateDirectory(outputPath);
            network.Save($"{outputPath}/model.model");
            File.WriteAllText($"{outputPath}/learning.json",Newtonsoft.Json.JsonConvert.SerializeObject(new {TrainingError=trainingError,TestingError=testingError}));
        }
        
        private static (Function network, LearningCurvesData learningData) LoadModel(string path,ModelFormat format)
        {
            var network=Function.Load(path,DeviceDescriptor.CPUDevice,format);
            var dirPath = Path.GetDirectoryName(path);
            var data = LearningCurvesData.LoadFrom(dirPath);
            return (network,data);
        }

        private static Function CreateTransferLearningModel(string baseModelFile,string hiddenNodeName,ParameterCloningMethod parameterCloningMethod=ParameterCloningMethod.Freeze)
        {
            Function baseModel = Function.Load(baseModelFile, DeviceDescriptor.CPUDevice);
            Function lastNode = baseModel.FindByName(hiddenNodeName);

            // Clone the desired layers with fixed weights
            Function clonedLayer = CNTKLib.AsComposite(lastNode).Clone(parameterCloningMethod);
            return clonedLayer;
        }
    
    }
}
