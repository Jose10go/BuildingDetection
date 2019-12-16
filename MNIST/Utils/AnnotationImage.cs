using BuildingDetection.Yolo;
using CNTKUtil;
using MNIST.Utils;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
namespace MNIST
{
    public class AnnotationImage : IData
    {
        public AnnotationImage()
        {
            Object = new List<YoloBoundingBox>();
        }
        private Bitmap Image => new Bitmap(Bitmap.FromFile(FileName)).Resize(Width, Height);
        public float[] Features => Image.ExtractCHW();
        public string Reference => FileName;


        public int Width { get; private set; }
        public int Height { get; private set; }
        public int S { get; private set; }
        public int B { get; private set; }
        public int C { get; private set; }


        public string FileName { get; private set; }

        public List<YoloBoundingBox> Object { get; set; }
        public static AnnotationImage FromFile(string annotationImagePath, string imagePath, int width, int height)
        {
            var result = new AnnotationImage
            {
                Width = width,
                Height = height,
                FileName = imagePath,
            };

            var lines = File.ReadLines(annotationImagePath);
            foreach (var line in lines)
            {
                var elements = line.Split(' ');
                var classId = int.Parse(elements[0]);
                result.Object.Add(new YoloBoundingBox
                {
                    Dimensions = new BoundingBoxDimensions()
                    {
                        X = float.Parse(elements[1]),
                        Y = float.Parse(elements[2]),
                        Width = float.Parse(elements[3]),
                        Height = float.Parse(elements[4]),
                    },
                    Confidence = 1,
                    Label = YoloBoundingBox.Tags[classId],
                    BoxColor = YoloBoundingBox.TagColors[classId]
                });
            }

            return result;
        }

    }
}
