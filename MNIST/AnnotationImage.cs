using CNTKUtil;
using MNIST.Utils;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;

namespace MNIST
{
    public class AnnotationImage:IData
    {
        public AnnotationImage()
        {
            Object = new List<AnnotationObject>();
        }
        private Bitmap Image => new Bitmap(Bitmap.FromFile(FileName)).Resize(Width, Height);

        public float[] Features => Image.ExtractCHW();
        public float[] Labels => ToOutput(C: 1); 

        public int Width { get; private set; }
        public int Height { get; private set; }
        public string FileName { get; private set; }

        public List<AnnotationObject> Object { get; set; }
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
                result.Object.Add(new AnnotationObject
                {
                    ClassId = int.Parse(elements[0]),
                    X = float.Parse(elements[1]),
                    Y = float.Parse(elements[2]),
                    W = float.Parse(elements[3]),
                    H = float.Parse(elements[4]), 
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
                var x = (int)(item.X * W);
                var y =(int)(item.Y * H);
                var row =(int)( y * S/(float)H);
                var column = (int)( x * S /(float)W);
                result[row*S + column*S + 0] = item.X;
                result[row*S + column*S + 1] = item.Y;
                result[row*S + column*S + 2] = item.W;
                result[row*S + column*S + 3] = item.H;
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

    public class AnnotationObject
    {
        public int ClassId { get; set; }

        public float X { get; set; }

        public float Y { get; set; }

        public float W { get; set; }
    
        public float H { get; set; }
    }
}
