using CNTK;
using CNTKUtil;
using Microsoft.ML;
using Microsoft.ML.Data;
using MNIST.Utils;
using System;
using System.Collections;
using System.IO;
using System.Linq;
using XPlot.Plotly;

namespace MNIST
{
    class Program
    {
        public static (Function,Variable,Variable) BuildYoloDNN(int S=7,int B=2,int C=20,int H=448,int W=448) 
        {
            //input
            var features = Variable.InputVariable(new int[] { H, W, 3 }, DataType.Float, "features");
            var labels = Variable.InputVariable(new int[] { S,S,B*5+C }, DataType.Float, "labels");

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
                                  .Dense(new[] {4096}, activation: leakyRelu)
                                  .Dense(new[] {S,S,B*5+C})
                                  .ToNetwork();

            return (network,features,labels);
        }
       
        static void Main(string[] args)
        {
            // create a machine learning context
            var context = new MLContext();
            // load data
            Console.WriteLine("Loading data....");

            var l = ReadAnnotationImages.ReadFromDirectory(@"C:\Users\Jose10go\Downloads\dataset", 448, 448);
            var training = l.Take(4 * l.Count / 5);
            var testing = l.Skip(4 * l.Count / 5);

            //// load training and testing data
            //var training = context.Data.CreateEnumerable<Digit>(trainDataView, reuseRowObject: false);
            //var testing = context.Data.CreateEnumerable<Digit>(testDataView, reuseRowObject: false);

            // set up data arrays
            var training_data = training.Select(v => v.Image.ExtractCHW()).ToArray();
            var training_labels = training.Select(v => v.ToOutput(C:1)).ToArray();
            var testing_data = testing.Select(v => v.Image.ExtractCHW()).ToArray();
            var testing_labels = testing.Select(v => v.ToOutput(C:1)).ToArray();

            var (network, features, labels) = BuildYoloDNN(C: 1);

            Console.WriteLine("Model architecture:");
            Console.WriteLine(network.ToSummary());

            // set up the loss function and the error function
            var errorFunc = CNTKLib.SquaredError(network.Output, labels);
            var lossFunc = CNTKLib.SquaredError(network.Output, labels);

            var maxEpochs = 135;
            var batchSize = 64;

            // train the model
            Console.WriteLine("Epoch\tTrain\tTrain\tTest");
            Console.WriteLine("\tLoss\tError\tError");
            Console.WriteLine("-----------------------------");

            var loss = new double[maxEpochs];
            var trainingError = new double[maxEpochs];
            var testingError = new double[maxEpochs];
            var batchCount = 0;

            for (int epoch = 0; epoch < maxEpochs; epoch++)
            {
                var learner = CNTKLib.MomentumSGDLearner(
                new ParameterVector((ICollection)network.Parameters()),
                new TrainingParameterScheduleDouble(GetLearningRate(epoch)),
                new TrainingParameterScheduleDouble(0.9));
                // train one epoch on batches
                loss[epoch] = 0.0;
                trainingError[epoch] = 0.0;
                batchCount = 0;
                training_data.Index().Shuffle().Batch(batchSize, (indices, begin, end) =>
                {
                    // get the current batch
                    var featureBatch = features.GetBatch(training_data, indices, begin, end);
                    var labelBatch = labels.GetBatch(training_labels, indices, begin, end);

                    var trainer = network.GetTrainer(learner, lossFunc, errorFunc);
                    // train the network on the batch
                    var result = trainer.TrainBatch(
                                        new[] {
                                            (features, featureBatch),
                                            (labels,  labelBatch)
                                        },
                                        false
                                    );
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
                testing_data.Batch(batchSize, (data, begin, end) =>
                {
                    // get the current batch for testing
                    var featureBatch = features.GetBatch(testing_data, begin, end);
                    var labelBatch = labels.GetBatch(testing_labels, begin, end);

                    var evaluator = network.GetEvaluator(errorFunc);
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
            }

            // show final results
            var finalError = testingError.Last();
            Console.WriteLine();
            Console.WriteLine($"Final test error: {finalError:0.00}");
            Console.WriteLine($"Final test accuracy: {1 - finalError:0.00}");

            network.Save("MODEL.MODEL");
        }

        private static double GetLearningRate(int epoch)
        {
            if (epoch < 75)
                return 1e2;
            if (epoch < 105)
                return 1e3;
            return 1e4;
        }
    }
}
