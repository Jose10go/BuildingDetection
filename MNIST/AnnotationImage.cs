using MNIST.Utils;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;

namespace MNIST
{
    public class AnnotationImage
    {
        public AnnotationImage()
        {
            Object = new List<AnnotationObject>();
        }

        public string FileName { get; set; }
        public Bitmap Image { get; set; }
        public List<AnnotationObject> Object { get; set; }
        public static AnnotationImage FromFile(string annotationImagePath, string imagePath, int width, int height)
        {
            var result = new AnnotationImage();
            result.FileName = imagePath;
            var lines = File.ReadLines(annotationImagePath);
            result.Image = new Bitmap(Bitmap.FromFile(imagePath)).Resize(width, height);
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

        public float[] ToOutput(int S = 7, int B = 2, int C = 20, int H = 448, int W = 448) 
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
