using CNTK;
using CNTKUtil;
using System;
using System.Collections;

namespace BuildingDetection.Yolo
{
    public static class YOLO
    {
        public static int S = 7;
        public static int B = 2;
        public static int C = 1; //20;
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
            return NetUtil.MeanSquaredError(network.Output, labels); ;
        }

        public static Function GetYoloErrorFunction(this Function network)
        {
            return NetUtil.MeanAbsoluteError(network.Output, labels);
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
