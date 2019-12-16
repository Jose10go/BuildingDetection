using BuildingDetection.TinyYoloV2;
using BuildingDetection.Utils;
using BuildingDetection.Yolo;
using CNTK;
using CNTKUtil;
using MNIST.Utils;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
namespace MNIST
{
    class Program
    {
        static void Main(string[] args)
        {
            //Examples
            //BuildAndTrainYOLO();
            //TestDrawer();
            //TestDrawLearningCurve();
            //TestDrawLearningCurve2();
            //EvaluateModel();
            //TransferLearning();
        }

        private static void TransferLearning()
        {
            var (training, testing) = LoadData();

            //build network
            var yolo2 = TinyYOLOV2.CreateTransferLearningModel<YOLO>(@"../../../../TinyYoloV2/yolo2.onnx","activation7");
            Console.WriteLine("Model architecture:");
            Console.WriteLine(yolo2.Network.ToSummary());
 
            yolo2.TrainMinibatchSource(training, testing, "yolo2", 100, 1);
        }

        private static void EvaluateModel()
        {
            var yolo = Model.LoadModel<YOLO>("model9.model",ModelFormat.CNTKv2);
            var (evaluationData, _) = LoadData(1);
            foreach (var item in evaluationData)
                yolo.Evaluate(item);

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
            var yolo = new YOLO();
            Console.WriteLine("Model architecture:");
            Console.WriteLine(yolo.Network.ToSummary());

            yolo.TrainMinibatchSource(training, testing, "yolo", 10, 3);
        }

        static (IData[] training,IData[] testing) LoadData(double percent = 0.8) 
        {
            Console.WriteLine("Loading data....");
            var l = ReadAnnotationImages.ReadFromDirectory(@"../../../../dataset",TinyYOLOV2.W,TinyYOLOV2.H);
            var split = (int)(percent * l.Count);
            var training = l.Take(split).ToArray();
            var testing = l.Skip(split).ToArray();
            return (training, testing);
        }

    }
}
