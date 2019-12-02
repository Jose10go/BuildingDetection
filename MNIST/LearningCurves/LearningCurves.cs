using BuildingDetection.LearningCurves;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;
using System.Collections.Generic;
using System.IO;

namespace BuildingDetection.Utils
{
    public class LearningCurves
    {
        public const string LearningCurvesFilename= "learningCurves.svg";
        PlotModel PlotModel;
        public LearningCurves(List<double> trainErrors, List<double> testErrors) : this(new LearningCurvesData() {TrainingError=trainErrors,TestingError=testErrors })
        {
        }
        public LearningCurves(LearningCurvesData curvesData):base()
        {
            this.PlotModel = new PlotModel();
            this.PlotModel.PlotType = PlotType.XY;
            this.PlotModel.Axes.Add(new LinearAxis() { Position=AxisPosition.Bottom,Key = "Horizontal", Title="Epochs"});
            this.PlotModel.Axes.Add(new LinearAxis() { Position = AxisPosition.Left,Maximum = 1,Minimum=0, Key = "Vertical",Title="Error" });
            var train = new FunctionSeries((x) => curvesData.TrainingError[(int)x], 0, curvesData.TrainingError.Count - 1, curvesData.TrainingError.Count, "Training Error");
            train.XAxisKey = "Horizontal";
            train.YAxisKey = "Vertical";
            this.PlotModel.Series.Add(train);
            var test = new FunctionSeries((x) => curvesData.TestingError[(int)x], 0, curvesData.TestingError.Count - 1, curvesData.TestingError.Count, "Testing Error");
            test.XAxisKey = "Horizontal";
            test.YAxisKey = "Vertical";
            this.PlotModel.Series.Add(test);
        }

        public void ExportSVG(string filename= LearningCurvesFilename, int Width=500,int Height=500) 
        {
            using (FileStream f = new FileStream(filename, FileMode.OpenOrCreate))
                    SvgExporter.Export(this.PlotModel, f, Width, Height,false);
        }

        public static void ExportSvgFromData(string dataPath, string filename = LearningCurvesFilename, int Width = 500, int Height = 500) 
        {
            var data=LearningCurvesData.LoadFrom(dataPath);
            LearningCurves learningCurves = new LearningCurves(data);
            learningCurves.ExportSVG(filename,Width,Height);
        }
    }
}
