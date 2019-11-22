using MNIST.Utils;
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
        public Bitmap Image { get; set; }
        public List<AnnotationObject> Object { get; set; }
        public static AnnotationImage FromFile(string annotationImagePath, string imagePath, int width, int height)
        {
            var result = new AnnotationImage();
            var lines = File.ReadLines(annotationImagePath);
            result.Image = new Bitmap(Bitmap.FromFile(imagePath)).Resize(width, height);
            foreach (var line in lines)
            {
                var elements = line.Split(' ');
                result.Object.Add(new AnnotationObject
                {
                    ClassId = int.Parse(elements[0]),
                    Xmin = double.Parse(elements[1]),
                    Xmax = double.Parse(elements[2]),
                    Ymin = double.Parse(elements[3]),
                    Ymax = double.Parse(elements[4])
                });
            }

            return result;
        }
    }

    public class AnnotationObject
    {
        public int ClassId { get; set; }

        public double Xmin { get; set; }

        public double Ymin { get; set; }

        public double Xmax { get; set; }

        public double Ymax { get; set; }
    }
}
