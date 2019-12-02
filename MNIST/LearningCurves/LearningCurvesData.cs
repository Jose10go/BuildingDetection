using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace BuildingDetection.LearningCurves
{
    public class LearningCurvesData
    {
        public const string LearningDataFileName = "learning.json";
        public List<double> TrainingError { get; set; } 
        public List<double> TestingError { get; set; }

        public static LearningCurvesData LoadFrom(string Path) 
        {
            var anonimous = new { TrainingError = new List<double>(), TestingError = new List<double>() };
            if (File.Exists(Path))
            {
                var str = File.ReadAllText(Path);
                return Newtonsoft.Json.JsonConvert.DeserializeObject<LearningCurvesData>(str);
            }
            if (Directory.Exists(Path))
            {
                var str = File.ReadAllText(System.IO.Path.Combine(Path, LearningDataFileName));
                return Newtonsoft.Json.JsonConvert.DeserializeObject<LearningCurvesData>(str);
            }
            return null;
        }
    }
}
