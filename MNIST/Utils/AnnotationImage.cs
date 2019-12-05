using BuildingDetection.Yolo;
using CNTKUtil;
using MNIST.Utils;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using static BuildingDetection.Yolo.YOLO;
namespace MNIST
{
    public class AnnotationImage:IData
    {
        public AnnotationImage()
        {
            Object = new List<YoloBoundingBox>();
        }
        private Bitmap Image => new Bitmap(Bitmap.FromFile(FileName)).Resize(W,H);

        public float[] Features => Image.ExtractCHW();
        public float[] Labels => ToOutput(C: 1);
        public string Reference => FileName;

        public int Width { get; private set; }
        public int Height { get; private set; }
        public string FileName { get; private set; }

        public List<YoloBoundingBox> Object { get; set; }
        public static AnnotationImage FromFile(string annotationImagePath, string imagePath, int width, int height)
        {
            var result = new AnnotationImage
            {
                Width = width,
                Height = height,
                FileName = imagePath
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
                    Label = YOLO.Tags[classId],
                    BoxColor=YOLO.TagColors[classId]
                });
            }

            return result;
        }

        private float[] ToOutput(int S = 7, int B = 2, int C = 20, int H = 448, int W = 448) 
        {
            //TODO: Improve THIS
            var result = new float[S*S*(B * 5 + C)];
            foreach (var item in Object)
            {
                var x = (int)(item.Dimensions.X * W);
                var y =(int)(item.Dimensions.Y * H);
                var row =(int)( y * S/(float)H);
                var column = (int)( x * S /(float)W);
                result[row*S + column*S + 0] = item.Dimensions.X;
                result[row*S + column*S + 1] = item.Dimensions.Y;
                result[row*S + column*S + 2] = item.Dimensions.Width;
                result[row*S + column*S + 3] = item.Dimensions.Height;
                result[row*S + column*S + 4] = 1;
                result[row*S + column*S + 5] = 0;
                result[row*S + column*S + 6] = 0;
                result[row*S + column*S + 7] = 0;
                result[row*S + column*S + 8] = 0;
                result[row*S + column*S + 9] = 0;
                result[row*S + column*S + 10] = 1;
            }
            return result;
        }
    }

}
