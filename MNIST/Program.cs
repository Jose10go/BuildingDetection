using CNTK;
using CNTKUtil;
using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections;
using System.IO;
using System.Linq;
using XPlot.Plotly;
namespace MNIST
{
    class Digit
    {
        [VectorType(784)]
        public float[] PixelValues = default;

        [LoadColumn(0)]
        public float Number = default;

        public float[] GetFeatures() => PixelValues;

        public float[] GetLabel() => new float[] {
            Number == 0 ? 1.0f : 0.0f,
            Number == 1 ? 1.0f : 0.0f,
            Number == 2 ? 1.0f : 0.0f,
            Number == 3 ? 1.0f : 0.0f,
            Number == 4 ? 1.0f : 0.0f,
            Number == 5 ? 1.0f : 0.0f,
            Number == 6 ? 1.0f : 0.0f,
            Number == 7 ? 1.0f : 0.0f,
            Number == 8 ? 1.0f : 0.0f,
            Number == 9 ? 1.0f : 0.0f,
        };
        
    }


    class Program
    {
        // filenames for data set
        private static string trainDataPath = Path.Combine(Environment.CurrentDirectory, "mnist_train.csv");
        private static string testDataPath = Path.Combine(Environment.CurrentDirectory, "mnist_test.csv");
    
        public static Function BuildYoloDNN(int S=7,int B=2,int C=20,int H=416,int W=448) 
        {
            //input
            var features = Variable.InputVariable(new int[] { 3, H, W }, DataType.Float, "features");
            var labels = Variable.InputVariable(new int[] { S,S,B*5+C }, DataType.Float, "labels");

            //buildNetwork alo LINQ
            Func<Variable, Function> leakyRelu = (Variable v) => CNTKLib.LeakyReLU(v, 0.1);
            var network = features.Convolution(new[] {7,7,64 },strides:new[] {2},activation: leakyRelu)
                                  .Pooling(PoolingType.Max,new[] { 2,2},new[] {2})
                                  .Convolution(new[] { 3,3,192}, activation: leakyRelu)
                                  .Pooling(PoolingType.Max,new[] { 2,2},new[] {2})
                                  .Convolution(new[] { 1, 1, 128 }, activation: leakyRelu)
                                  .Convolution(new[] { 3, 3, 256 }, activation: leakyRelu)
                                  .Convolution(new[] { 1, 1, 256 }, activation: leakyRelu)
                                  .Convolution(new[] { 3, 3, 512 }, activation: leakyRelu)
                                  .Pooling(PoolingType.Max, new[] { 2, 2 }, new[] { 2 })
                                  .Convolution(new[] { 1, 1, 256 }, activation: leakyRelu)
                                  .Convolution(new[] { 3, 3, 512 }, activation: leakyRelu)
                                  .Convolution(new[] { 1, 1, 256 }, activation: leakyRelu)
                                  .Convolution(new[] { 3, 3, 512 }, activation: leakyRelu)
                                  .Convolution(new[] { 1, 1, 256 }, activation: leakyRelu)
                                  .Convolution(new[] { 3, 3, 512 }, activation: leakyRelu)
                                  .Convolution(new[] { 1, 1, 256 }, activation: leakyRelu)
                                  .Convolution(new[] { 3, 3, 512 }, activation: leakyRelu)
                                  .Convolution(new[] { 1, 1, 512 }, activation: leakyRelu)
                                  .Convolution(new[] { 3, 3, 1024 }, activation: leakyRelu)
                                  .Pooling(PoolingType.Max, new[] { 2, 2 }, new[] { 2 })
                                  .Convolution(new[] { 1, 1, 512 }, activation: leakyRelu)
                                  .Convolution(new[] { 3, 3, 1024 }, activation: leakyRelu)
                                  .Convolution(new[] { 1, 1, 512 }, activation: leakyRelu)
                                  .Convolution(new[] { 3, 3, 1024 }, activation: leakyRelu)
                                  .Convolution(new[] { 3, 3, 1024 }, activation: leakyRelu)
                                  .Convolution(new[] { 3, 3, 1024 },strides:new[] {2}, activation: leakyRelu)
                                  .Convolution(new[] { 3, 3, 1024 }, activation: leakyRelu)
                                  .Convolution(new[] { 3, 3, 1024 }, activation: leakyRelu)
                                  .Dense(new[] {4096}, activation: leakyRelu)
                                  .Dense(new[] {S,S,B*5+C})
                                  .ToNetwork();

            return network;
        }

        static void Main(string[] args)
        {
            // create a machine learning context
            var context = new MLContext();
            // load data
            Console.WriteLine("Loading data....");
            var columnDef = new TextLoader.Column[]
            {
                new TextLoader.Column(nameof(Digit.PixelValues), DataKind.Single, 1, 784),
                new TextLoader.Column(nameof(Digit.Number), DataKind.Single, 0)
            };

            var trainDataView = context.Data.LoadFromTextFile(
                path: trainDataPath,
                columns: columnDef,
                hasHeader: true,
                separatorChar: ',');
            var testDataView = context.Data.LoadFromTextFile(
                path: testDataPath,
                columns: columnDef,
                hasHeader: true,
                separatorChar: ',');

            //// load training and testing data
            var training = context.Data.CreateEnumerable<Digit>(trainDataView, reuseRowObject: false);
            var testing = context.Data.CreateEnumerable<Digit>(testDataView, reuseRowObject: false);

            // set up data arrays
            var training_data = training.Select(v => v.GetFeatures()).ToArray();
            var training_labels = training.Select(v => v.GetLabel()).ToArray();
            var testing_data = testing.Select(v => v.GetFeatures()).ToArray();
            var testing_labels = testing.Select(v => v.GetLabel()).ToArray();
           
            //input
            var features = Variable.InputVariable(new int[] { }, DataType.Float, "features");
            var labels = Variable.InputVariable(new int[] { }, DataType.Float, "labels");

            var network = BuildYoloDNN(C:1);

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
                // train one epoch on batches
                loss[epoch] = 0.0;
                trainingError[epoch] = 0.0;
                batchCount = 0;
                training_data.Index().Shuffle().Batch(batchSize, (indices, begin, end) =>
                {
                    // get the current batch
                    var featureBatch = features.GetBatch(training_data, indices, begin, end);
                    var labelBatch = labels.GetBatch(training_labels, indices, begin, end);

                    var learner=CNTKLib.MomentumSGDLearner(
                    new ParameterVector((ICollection)network.Parameters()),
                    new TrainingParameterScheduleDouble(GetLearningRate(epoch)),
                    new TrainingParameterScheduleDouble(0.9));

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

            // plot the error graph
            var chart = Chart.Plot(
                new[]
                {
                    new Graph.Scatter()
                    {
                        x = Enumerable.Range(0, maxEpochs).ToArray(),
                        y = trainingError,
                        name = "training",
                        mode = "lines+markers"
                    },
                    new Graph.Scatter()
                    {
                        x = Enumerable.Range(0, maxEpochs).ToArray(),
                        y = testingError,
                        name = "testing",
                        mode = "lines+markers"
                    }
                });
            chart.WithXTitle("Epoch");
            chart.WithYTitle("Classification error");
            chart.WithTitle("Digit Training");

            // save chart
            File.WriteAllText("chart.html", chart.GetHtml());
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
