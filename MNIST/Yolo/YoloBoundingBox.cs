using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.IO;
using System.Text;

namespace BuildingDetection.Yolo
{
    public class YoloBoundingBox
    {
        public static string[] Tags = new[] { "building" };

        public static Color[] TagColors = new[] { Color.Red };
        public BoundingBoxDimensions Dimensions { get; set; }
        public string Label { get; set; }
        public float Confidence { get; set; }
        public RectangleF Rect
        {
            get { return new RectangleF(Dimensions.X, Dimensions.Y, Dimensions.Width,Dimensions.Height); }
        }
        public Color BoxColor { get; set; }

        public static void DrawBoundingBox(string inputImageLocation, string outputImageLocation, string imageName, IList<YoloBoundingBox> filteredBoundingBoxes)
        {
            Image image = Image.FromFile(Path.Combine(inputImageLocation, imageName));
            var originalImageHeight = image.Height;
            var originalImageWidth = image.Width;
            foreach (var box in filteredBoundingBoxes)
            {
                //var x = (uint)Math.Max(box.Dimensions.X, 0);
                //var y = (uint)Math.Max(box.Dimensions.Y, 0);
                //var width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
                //var height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);

                //x = (uint)originalImageWidth * x / (uint)YOLO.W;
                //y = (uint)originalImageHeight * y / (uint)YOLO.H;
                //width = (uint)originalImageWidth * width / (uint)YOLO.W;
                //height = (uint)originalImageHeight * height / (uint)YOLO.H;

                var width = originalImageWidth * box.Dimensions.Width;
                var height = originalImageHeight * box.Dimensions.Height;
                var x = box.Dimensions.X * originalImageWidth-width/2;
                var y = box.Dimensions.Y * originalImageHeight-height/2;

                string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";

                using (Graphics thumbnailGraphic = Graphics.FromImage(image))
                {
                    thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
                    thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
                    thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

                    // Define Text Options
                    Font drawFont = new Font("Arial", 12, FontStyle.Bold);
                    SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
                    SolidBrush fontBrush = new SolidBrush(Color.Black);
                    Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);
                    // Define BoundingBox options
                    Pen pen = new Pen(box.BoxColor, 3.2f);
                    SolidBrush colorBrush = new SolidBrush(box.BoxColor);

                    thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width,(int)size.Height);
                    thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);
                    thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
                }
            }
            if (!Directory.Exists(outputImageLocation))
            {
                Directory.CreateDirectory(outputImageLocation);
            }
            image.Save(Path.Combine(outputImageLocation, imageName));
        }
    }

    public class BoundingBoxDimensions
    {
        public float X { get; set; }
        public float Y { get; set; }
        public float Height { get; set; }
        public float Width { get; set; }
    }
}
