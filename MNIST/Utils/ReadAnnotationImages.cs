using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Xml.Serialization;

namespace MNIST.Utils
{
    static class ReadAnnotationImages
    {
        public static List<AnnotationImage> ReadFromDirectory(string path, int width, int height)
        {
            var annotationImages = new List<AnnotationImage>();
            var groupFiles = Directory.EnumerateFiles(path)
                .Select(file => (fileName: Path.GetFileNameWithoutExtension(file), ext: Path.GetExtension(file)))
                .GroupBy(file => file.fileName)
                .Where(groupFile => groupFile.Take(3).Count() == 2);

            foreach (var groupFile in groupFiles)
            {
                var annotationFile = groupFile.Single(x => x.ext == ".txt");
                var imageFile = groupFile.Single(x => x.ext == ".jpg");
                var annotationFilePath = Path.Combine(path, $"{annotationFile.fileName}{annotationFile.ext}");
                var imageFilePath = Path.Combine(path, $"{imageFile.fileName}{imageFile.ext}");
                annotationImages.Add(AnnotationImage.FromFile(annotationFilePath, imageFilePath, width, height));
            }

            return annotationImages;
        }
    }
}
