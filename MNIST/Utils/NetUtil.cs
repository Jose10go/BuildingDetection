using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace CNTKUtil
{
    /// <summary>
    /// A collection of utilities to work with neural networks.
    /// </summary>
    public static class NetUtil
    {
        // the current compute device
        private static CNTK.DeviceDescriptor currentDevice = null;

        /// <summary>
        /// Get the current compute device. The library will use the first GPU
        /// it finds, and default to the CPU if no GPUs are found.
        /// </summary>
        /// <returns>The current compute device.</returns>
        public static CNTK.DeviceDescriptor CurrentDevice
        {
            get
            {
                if (currentDevice == null)
                {
                    currentDevice = CNTK.DeviceDescriptor.CPUDevice;
                    foreach (var gpuDevice in CNTK.DeviceDescriptor.AllDevices())
                    {
                        if (gpuDevice.Type == CNTK.DeviceKind.GPU)
                        {
                            currentDevice = gpuDevice;
                            break;
                        }
                    }
                }
                return currentDevice;
            }
        }

        /// <summary>
        /// Create an input/output variable for a neural network.
        /// </summary>
        /// <param name="shape">The shape of the variable.</param>
        /// <param name="dataType">The data type of the variable.</param>
        /// <returns>The created neural network variable.</returns>
        public static CNTK.Variable Var(
            IEnumerable<int> shape,
            CNTK.DataType dataType,
            string name = "",
            List<CNTK.Axis> dynamicAxes = null)
        {
            return CNTK.Variable.InputVariable(
                CNTK.NDShape.CreateNDShape(shape), 
                dataType,
                name: name,
                dynamicAxes: dynamicAxes);
        }

        public static CNTK.Variable Dense(
            this CNTK.Variable input,
            int[] outputDim,
            Func<CNTK.Variable, CNTK.Function> activation,
            string outputName = "")
        {
            return activation(Dense(input, outputDim,outputName));
        }

        public static CNTK.Variable Dense(
            this CNTK.Variable input,
            int[] outputDim,
            string outputName = "")
        {
            var shape = CNTK.NDShape.CreateNDShape(outputDim.ToList().Append(CNTK.NDShape.InferredDimension));
            var timesParam = new CNTK.Parameter(
                shape,
                CNTK.DataType.Float,
                CNTK.CNTKLib.GlorotUniformInitializer(
                    CNTK.CNTKLib.DefaultParamInitScale,
                    CNTK.CNTKLib.SentinelValueForInferParamInitRank,
                    CNTK.CNTKLib.SentinelValueForInferParamInitRank, 1),
                CurrentDevice,
                "timesParam_" + outputName);
            var timesFunction = CNTK.CNTKLib.Times(timesParam,input,(uint)outputDim.Length,0);
            var plusParam = new CNTK.Parameter(
                CNTK.NDShape.CreateNDShape(new int[] { CNTK.NDShape.InferredDimension }),
                0.0f,
                CurrentDevice,
                "plusParam_" + outputName);
            var result = CNTK.CNTKLib.Plus(plusParam, timesFunction, outputName);
            return result;
        }

        public static CNTK.Variable Convolution2D(
             this CNTK.Variable input,
             int outputChannels,
             int[] filterShape,
             bool padding = true,
             bool bias = true,
             int[] strides = null,
             Func<CNTK.Variable, CNTK.Function> activation = null,
             string outputName = "")
        {
            var convolution_map_size = new int[] {
                filterShape[0],
                filterShape[1],
                CNTK.NDShape.InferredDimension,
                outputChannels
            };
            if (strides == null)
            {
                strides = new int[] { 1 };
            }
            return Convolution(convolution_map_size, input, padding, bias, strides, activation, outputName);
        }

        static public CNTK.Variable ConvolutionTranspose(
            this CNTK.Variable input,
            int[] filterShape,
            int numberOfFilters,
            Func<CNTK.Variable, CNTK.Function> activation = null,
            bool padding = true,
            int[] strides = null,
            bool bias = true,
            int[] outputShape = null,
            uint reductionRank = 1,
            int[] dilation = null,
            uint maxTempMemSizeInSamples = 0,
            string name = "")
        {
            if (strides == null)
            {
                strides = new int[] { 1 };
            }
            var sharing = PadToShape(filterShape, true);
            var thePadding = PadToShape(filterShape, padding);
            if (reductionRank != 1)
            {
                throw new NotSupportedException("reduction_rank should be 1");
            }
            thePadding = ConcatenateArrays(thePadding, new bool[] { false });
            if (dilation == null)
            {
                dilation = PadToShape(filterShape, 1);
            }
            var output_channels_shape = new int[] { numberOfFilters };
            var kernel_shape = ConcatenateArrays(filterShape, output_channels_shape, new int[] { CNTK.NDShape.InferredDimension });
            var output_full_shape = outputShape;
            if (output_full_shape != null)
            {
                output_full_shape = ConcatenateArrays(outputShape, output_channels_shape);
            }
            var filter_rank = filterShape.Length;
            var init = CNTK.CNTKLib.GlorotUniformInitializer(CNTK.CNTKLib.DefaultParamInitScale, CNTK.CNTKLib.SentinelValueForInferParamInitRank, CNTK.CNTKLib.SentinelValueForInferParamInitRank, 1);
            var W = new CNTK.Parameter(kernel_shape, CNTK.DataType.Float, init, NetUtil.CurrentDevice, name = "W");
            var r = CNTK.CNTKLib.ConvolutionTranspose(
              convolutionMap: W,
              operand: input,
              strides: strides,
              sharing: new CNTK.BoolVector(sharing),
              autoPadding: new CNTK.BoolVector(thePadding),
              outputShape: output_full_shape,
              dilation: dilation,
              reductionRank: reductionRank,
              maxTempMemSizeInSamples: maxTempMemSizeInSamples);
            if (bias)
            {
                var b_shape = ConcatenateArrays(MakeOnesArray(filterShape.Length), output_channels_shape);
                var b = new CNTK.Parameter(b_shape, 0.0f, NetUtil.CurrentDevice, "B");
                r = CNTK.CNTKLib.Plus(r, b);
            }
            if (activation != null)
            {
                r = activation(r);
            }
            return r;
        }

        public static CNTK.Variable TransposeAxes(
            this CNTK.Variable input,
            CNTK.Axis axis1,
            CNTK.Axis axis2)
        {
            return CNTK.CNTKLib.TransposeAxes(input, axis1, axis2);
        }

        public static CNTK.Variable Pooling(
            this CNTK.Variable input,
            CNTK.PoolingType poolingType,
            int[] windowShape,
            int[] strides)
        {
            return CNTK.CNTKLib.Pooling(input, poolingType, windowShape, strides,new[] {true});
        }

        public static CNTK.Variable Pooling(
            this CNTK.Variable input,
            CNTK.PoolingType poolingType,
            CNTK.NDShape windowShape,
            int[] strides)
        {
            return CNTK.CNTKLib.Pooling(input, poolingType, windowShape, strides);
        }

        public static CNTK.Variable Dropout(
            this CNTK.Variable input,
            double dropoutRate)
        {
            return CNTK.CNTKLib.Dropout(input, 0.5);
        }

        public static CNTK.Variable OneHotOp(
            this CNTK.Variable input,
            int numberOfClasses,
            bool outputSparse
            )
        {
            return CNTK.CNTKLib.OneHotOp(input, (uint)numberOfClasses, outputSparse, new CNTK.Axis(0));
        }

        public static CNTK.Variable Embedding(
            this CNTK.Variable input,
            int embeddingDimensions)
        {
            var weight_shape = new int[] { embeddingDimensions, CNTK.NDShape.InferredDimension };
            var E = new CNTK.Parameter(
              weight_shape,
              CNTK.DataType.Float,
              CNTK.CNTKLib.GlorotUniformInitializer(),
              NetUtil.CurrentDevice);

            return CNTK.CNTKLib.Times(E, input);
        }

        public static CNTK.Variable MultiplyBy<T>(
            this CNTK.Variable input,
            T scalar)
        {
            var scalarTensor = CNTK.Constant.Scalar<T>(scalar, NetUtil.CurrentDevice);
            return CNTK.CNTKLib.ElementTimes(scalarTensor, input);
        }

        public static CNTK.Variable Reshape(
            this CNTK.Variable input,
            CNTK.NDShape newShape)
        {
            return CNTK.CNTKLib.Reshape(input, newShape);
        }

        public static CNTK.Function ToNetwork(
            this CNTK.Variable input)
        {
            return input;
        }

        public static string ToSummary(this CNTK.Function model)
        {
            var sb = new StringBuilder();
            sb.AppendFormat("\tInput = " + model.Arguments[0].Shape.AsString());
            sb.Append(Environment.NewLine);
            for (int i = 0; i < model.Outputs.Count; i++)
            {
                sb.AppendFormat("\tOutput = " + model.Outputs[i].Shape.AsString());
                sb.Append(Environment.NewLine);
            }
            sb.Append(Environment.NewLine);

            var numParameters = 0;
            foreach (var x in model.Parameters())
            {
                var shape = x.Shape;
                var p = shape.TotalSize;
                sb.AppendFormat(string.Format("\tFilter Shape:{0,-30} Params:{1}", shape.AsString(), p));
                sb.Append(Environment.NewLine);
                numParameters += p;
            }
            sb.AppendFormat(string.Format("\tTotal Number of Parameters: {0:N0}", numParameters));
            sb.Append(Environment.NewLine);
            return sb.ToString();
        }

        public static CNTK.Function BinaryClassificationError(CNTK.Variable prediction, CNTK.Variable labels)
        {
            var round_predictions = CNTK.CNTKLib.Round(prediction);
            var equal_elements = CNTK.CNTKLib.NotEqual(round_predictions, labels);
            var result = CNTK.CNTKLib.ReduceMean(equal_elements, CNTK.Axis.AllStaticAxes());
            return result;
        }

        public static CNTK.Function MeanSquaredError(CNTK.Variable prediction, CNTK.Variable labels)
        {
            var squared_errors = CNTK.CNTKLib.Square(CNTK.CNTKLib.Minus(prediction, labels));
            var result = CNTK.CNTKLib.ReduceMean(squared_errors,CNTK.Axis.AllStaticAxes()); // TODO -- allStaticAxes?
            return result;
        }

        public static CNTK.Function MeanAbsoluteError(CNTK.Variable prediction, CNTK.Variable labels)
        {
            var absolute_errors = CNTK.CNTKLib.Abs(CNTK.CNTKLib.Minus(prediction, labels));
            var result = CNTK.CNTKLib.ReduceMean(absolute_errors, CNTK.Axis.AllStaticAxes()); // TODO -- allStaticAxes? 
            return result;
        }


        /// <summary>
        /// Get an RMSProp learner to train the network.
        /// </summary>
        /// <param name="input">The network to train.</param>
        /// <param name="learningRateSchedule">The learning rate schedule.</param>
        /// <param name="gamma">The gamma value.</param>
        /// <param name="inc">The inc value.</param>
        /// <param name="dec">The dec value.</param>
        /// <param name="max">The max value.</param>
        /// <param name="min">The min value.</param>
        /// <returns>An RMSProp learner to train the network.</returns>
        public static CNTK.Learner GetRMSPropLearner(
            this CNTK.Function input,
            double learningRateSchedule,
            double gamma,
            double inc,
            double dec,
            double max,
            double min)
        {
            var parameterVector = new CNTK.ParameterVector((System.Collections.ICollection)input.Parameters());
            return CNTK.CNTKLib.RMSPropLearner(
                parameterVector,
                new CNTK.TrainingParameterScheduleDouble(learningRateSchedule),
                gamma,
                inc,
                dec,
                max,
                min);
        }

        /// <summary>
        /// Get an Adam learner to train the network.
        /// </summary>
        /// <param name="input">The network to train.</param>
        /// <param name="learningRateSchedule">The learning rate schedule.</param>
        /// <param name="momentumSchedule">The moment schedule.</param>
        /// <param name="unitGain">The unit gain.</param>
        /// <returns>An Adamlearner to train the network.</returns>
        public static CNTK.Learner GetAdamLearner(
            this CNTK.Function input,
            (double, uint) learningRateSchedule,
            (double, uint) momentumSchedule,
            bool unitGain = true)
        {
            var parameterVector = new CNTK.ParameterVector((System.Collections.ICollection)input.Parameters());
            return CNTK.CNTKLib.AdamLearner(
                parameterVector,
                new CNTK.TrainingParameterScheduleDouble(learningRateSchedule.Item1, learningRateSchedule.Item2),
                new CNTK.TrainingParameterScheduleDouble(momentumSchedule.Item1, momentumSchedule.Item2),
                unitGain);
        }

        /// <summary>
        /// Get an Adam learner to train the network.
        /// </summary>
        /// <param name="input">The network to train.</param>
        /// <param name="learningRateSchedule">The learning rate schedule.</param>
        /// <param name="momentumSchedule">The moment schedule.</param>
        /// <param name="unitGain">The unit gain.</param>
        /// <returns>An Adamlearner to train the network.</returns>
        public static CNTK.Learner GetAdamLearner(
            this CNTK.Function input,
            double learningRateSchedule,
            double momentumSchedule,
            bool unitGain = true)
        {
            var parameterVector = new CNTK.ParameterVector((System.Collections.ICollection)input.Parameters());
            return CNTK.CNTKLib.AdamLearner(
                parameterVector,
                new CNTK.TrainingParameterScheduleDouble(learningRateSchedule),
                new CNTK.TrainingParameterScheduleDouble(momentumSchedule),
                unitGain);
        }

        /// <summary>
        /// Get an Ada Delta learner to train the network.
        /// </summary>
        /// <param name="input">The network to train.</param>
        /// <param name="learningRateSchedule">The learning rate schedule.</param>
        /// <returns>An AdaDeltaLearner to train the network.</returns>
        public static CNTK.Learner GetAdaDeltaLearner(
            this CNTK.Function input,
            double learningRateSchedule
            )
        {
            return CNTK.CNTKLib.AdaDeltaLearner(
                parameters: new CNTK.ParameterVector((System.Collections.ICollection)input.Parameters()),
                learningRateSchedule: new CNTK.TrainingParameterScheduleDouble(learningRateSchedule));
        }

        /// <summary>
        /// Get a trainer to train the network.
        /// </summary>
        /// <param name="input">The network to train.</param>
        /// <param name="lossFunc">The loss function to use.</param>
        /// <param name="evaluationFunc">The evaluation function to use.</param>
        /// <param name="learner">The learner to use.</param>
        /// <returns>A new trainer instance to train the network.</returns>
        public static CNTK.Trainer GetTrainer(
            this CNTK.Function input,
            CNTK.Learner learner,
            CNTK.Function lossFunc,
            CNTK.Function evaluationFunc)
        {
            return CNTK.CNTKLib.CreateTrainer(
                input, lossFunc,
                evaluationFunc,
                new CNTK.LearnerVector() { learner });
        }

        /// <summary>
        /// Get an evaluator to test the network.
        /// </summary>
        /// <param name="input">The network to test.</param>
        /// <param name="testFunc">The test function to use.</param>
        /// <returns>A new evaluator instance to test the network.</returns>
        public static CNTK.Evaluator GetEvaluator(
            this CNTK.Function input,
            CNTK.Function testFunc)
        {
            return CNTK.CNTKLib.CreateEvaluator(testFunc);
        }

        /// <summary>
        /// Train the network trainer on a batch.
        /// </summary>
        /// <param name="trainer">The trainer to use.</param>
        /// <param name="batch">The batch of features and labels to use.</param>
        /// <returns>A tuple of the current loss and evaluation values.</returns>
        public static (double Loss, double Evaluation) TrainBatch(
            this CNTK.Trainer trainer,
            (CNTK.Variable, CNTK.Value)[] batch,
            bool isSweepEndInArguments)
        {
            var dict = new Dictionary<CNTK.Variable, CNTK.Value>();
            foreach (var t in batch)
                dict.Add(t.Item1, t.Item2);

            trainer.TrainMinibatch(
                dict,
                false,
                CurrentDevice);

            return (
                Loss: trainer.PreviousMinibatchLossAverage(),
                Evaluation: trainer.PreviousMinibatchEvaluationAverage());
        }

        /// <summary>
        /// Train the network trainer on a batch.
        /// </summary>
        /// <param name="trainer">The trainer to use.</param>
        /// <param name="batch">The batch of features and labels to use.</param>
        /// <returns>A tuple of the current loss and evaluation values.</returns>
        public static (double Loss, double Evaluation) TrainBatch(
            this CNTK.Trainer trainer,
            (CNTK.Variable, CNTK.MinibatchData)[] batch)
        {
            var dict = new Dictionary<CNTK.Variable, CNTK.MinibatchData>();
            foreach (var t in batch)
                dict.Add(t.Item1, t.Item2);

            trainer.TrainMinibatch(
                dict,
                CurrentDevice);

            return (
                Loss: trainer.PreviousMinibatchLossAverage(),
                Evaluation: trainer.PreviousMinibatchEvaluationAverage());
        }


        /// <summary>
        /// Test the network evaluator on a batch.
        /// </summary>
        /// <param name="trainer">The evaluator to use.</param>
        /// <param name="batch">The batch of features and labels to use.</param>
        /// <returns>The current accuracy of the network.</returns>
        public static double TestBatch(
            this CNTK.Evaluator evaluator,
            (CNTK.Variable, CNTK.Value)[] batch)
        {
            var dict = new CNTK.UnorderedMapVariableValuePtr();
            foreach (var t in batch)
                dict.Add(t.Item1, t.Item2);

            return evaluator.TestMinibatch(
                dict,
                CurrentDevice);
        }

        /// <summary>
        /// Test the network evaluator on a batch.
        /// </summary>
        /// <param name="trainer">The evaluator to use.</param>
        /// <param name="batch">The batch of features and labels to use.</param>
        /// <returns>The current accuracy of the network.</returns>
        public static double TestBatch(
            this CNTK.Evaluator evaluator,
            (CNTK.Variable, CNTK.MinibatchData)[] batch)
        {
            var dict = new CNTK.UnorderedMapVariableMinibatchData();
            foreach (var t in batch)
                dict.Add(t.Item1, t.Item2);

            return evaluator.TestMinibatch(
                dict,
                CurrentDevice);
        }

        public static CNTK.Function Convolution(
          this CNTK.Variable input,
          int[] convolutionMapSize,
          bool padding,
          bool bias,
          int[] strides,
          Func<CNTK.Variable, CNTK.Function> activation = null,
          string outputName = "")
        {
            var W = new CNTK.Parameter(
              CNTK.NDShape.CreateNDShape(convolutionMapSize),
              CNTK.DataType.Float,
              CNTK.CNTKLib.GlorotUniformInitializer(
                  CNTK.CNTKLib.DefaultParamInitScale, 
                  CNTK.CNTKLib.SentinelValueForInferParamInitRank, 
                  CNTK.CNTKLib.SentinelValueForInferParamInitRank, 1),
              NetUtil.CurrentDevice, outputName + "_W");

            var result = CNTK.CNTKLib.Convolution(
                W, 
                input, 
                strides, 
                new CNTK.BoolVector(new bool[] { true }) /* sharing */, 
                new CNTK.BoolVector(new bool[] { padding }));

            if (bias)
            {
                var num_output_channels = convolutionMapSize[convolutionMapSize.Length - 1];
                var b_shape = ConcatenateArrays(MakeOnesArray(convolutionMapSize.Length - 2), new int[] { num_output_channels });
                var b = new CNTK.Parameter(b_shape, 0.0f, NetUtil.CurrentDevice, outputName + "_b");
                result = CNTK.CNTKLib.Plus(result, b);
            }

            if (activation != null)
            {
                result = activation(result);
            }
            return result;
        }

        /// <summary>
        /// Concatenate arrays.
        /// </summary>
        /// <typeparam name="T">The type of array element</typeparam>
        /// <param name="arguments">The arrays to concatenate</param>
        /// <returns>The concatenated array</returns>
        private static T[] ConcatenateArrays<T>(params T[][] arguments) where T : struct
        {
            var list = new List<T>();
            for (int i = 0; i < arguments.Length; i++)
            {
                list.AddRange(arguments[i]);
            }
            return list.ToArray();
        }

        /// <summary>
        /// Create an array filled with ones.
        /// </summary>
        /// <param name="numOnes">The number of ones to create</param>
        /// <returns>A new array filled with the specified number of ones</returns>
        private static int[] MakeOnesArray(int numOnes)
        {
            var ones = new int[numOnes];
            for (int i = 0; i < numOnes; i++)
            {
                ones[i] = 1;
            }
            return ones;
        }

        /// <summary>
        /// Create an array with the same size as a given array filled with default values.
        /// </summary>
        /// <typeparam name="T">The type of the value</typeparam>
        /// <param name="filter_shape">The array to use</param>
        /// <param name="value">The default value to use</param>
        /// <returns>A new array of the same size as filter_shape filled with the default values/returns>
        static T[] PadToShape<T>(int[] filter_shape, T value) where T : struct
        {
            var result = new T[filter_shape.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = value;
            }
            return result;
        }


    }
}
