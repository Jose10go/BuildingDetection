using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;
using static BuildingDetection.Yolo.YOLO;
namespace BuildingDetection.Yolo
{
    public class YOLOOutputParser
    {
        private static float Sigmoid(float value)
        {
            var k = (float)Math.Exp(value);
            return k / (1.0f + k);
        }

        private static float[] Softmax(float[] values)
        {
            var maxVal = values.Max();
            var exp = values.Select(v => Math.Exp(v - maxVal));
            var sumExp = exp.Sum();
            return exp.Select(v => (float)(v / sumExp)).ToArray();
        }

        private static int GetOffset(int x, int y, int channel)
        {
            return (x*S) + (y*S) + channel;
        }

        private static BoundingBoxDimensions ExtractBoundingBoxDimensions(float[] modelOutput, int x, int y, int channel)
        {
            return new BoundingBoxDimensions
            {
                X = modelOutput[GetOffset(x, y, channel)],
                Y = modelOutput[GetOffset(x, y, channel + 1)],
                Width = modelOutput[GetOffset(x, y, channel + 2)],
                Height = modelOutput[GetOffset(x, y, channel + 3)]
            };
        }

        private static float GetConfidence(float[] modelOutput, int x, int y, int channel)
        {
            return Sigmoid(modelOutput[GetOffset(x, y, channel + 4)]);
        }

        private static float[] ExtractClasses(float[] modelOutput, int x, int y)
        {
            float[] predictedClasses = new float[C];
            int predictedClassOffset = B*5;
            for (int predictedClass = 0; predictedClass < C; predictedClass++)
            {
                predictedClasses[predictedClass] = modelOutput[GetOffset(x, y, predictedClass + predictedClassOffset)];
            }
            return Softmax(predictedClasses);
        }

        private static (float, int) GetTopResult(float[] predictedClasses)
        {
            return predictedClasses.Select((predictedClass, index) => (Value: predictedClass, Index: index))
                                   .Max();         
        }

        private static float IntersectionOverUnion(RectangleF boundingBoxA, RectangleF boundingBoxB)
        {
            var areaA = boundingBoxA.Width * boundingBoxA.Height;
            if (areaA <= 0)
                return 0;
            var areaB = boundingBoxB.Width * boundingBoxB.Height;
            if (areaB <= 0)
                return 0;
            var minX = Math.Max(boundingBoxA.Left, boundingBoxB.Left);
            var minY = Math.Max(boundingBoxA.Top, boundingBoxB.Top);
            var maxX = Math.Min(boundingBoxA.Right, boundingBoxB.Right);
            var maxY = Math.Min(boundingBoxA.Bottom, boundingBoxB.Bottom);
            var intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);
            return intersectionArea / (areaA + areaB - intersectionArea);
        }
        
        public static IList<YoloBoundingBox> ParseOutputs(float[] yoloModelOutputs, float threshold = .3F)
        {
            var boxes = new List<YoloBoundingBox>();
            for (int row = 0; row < S; row++)
            {
                for (int column = 0; column < S; column++)
                {
                    for (int box = 0; box < B; box++)
                    {
                        var channel = (box * (5));
                        BoundingBoxDimensions boundingBoxDimensions = ExtractBoundingBoxDimensions(yoloModelOutputs, row, column, channel);
                        float confidence = GetConfidence(yoloModelOutputs, row, column, channel);
                        if (confidence < threshold)
                            continue;
                        float[] predictedClasses = ExtractClasses(yoloModelOutputs, row, column);
                        var (topResultScore, topResultIndex) = GetTopResult(predictedClasses);
                        var topScore = topResultScore * confidence;
                        if (topScore < threshold)
                            continue;
                        boxes.Add(new YoloBoundingBox()
                        {
                            Dimensions = boundingBoxDimensions,
                            Confidence = topScore,
                            Label = YoloBoundingBox.Tags[topResultIndex],
                            BoxColor = YoloBoundingBox.TagColors[topResultIndex]
                        });
                    }
                }
            }

            return boxes;
        }

        public static IList<YoloBoundingBox> FilterBoundingBoxes(IList<YoloBoundingBox> boxes, int limit, float threshold)
        {
            var activeCount = boxes.Count;
            var isActiveBoxes = new bool[boxes.Count];
            for (int i = 0; i < isActiveBoxes.Length; i++)
                isActiveBoxes[i] = true;
            var sortedBoxes = boxes.Select((b, i) => new { Box = b, Index = i })
                                   .OrderByDescending(b => b.Box.Confidence)
                                   .ToList();
            var results = new List<YoloBoundingBox>();
            for (int i = 0; i < boxes.Count; i++)
            {
                if (isActiveBoxes[i])
                {
                    var boxA = sortedBoxes[i].Box;
                    results.Add(boxA);
                    if (results.Count >= limit)
                        break;
                    for (var j = i + 1; j < boxes.Count; j++)
                    {
                        if (isActiveBoxes[j])
                        {
                            var boxB = sortedBoxes[j].Box;
                            if (IntersectionOverUnion(boxA.Rect, boxB.Rect) > threshold)
                            {
                                isActiveBoxes[j] = false;
                                activeCount--;
                                if (activeCount <= 0)
                                    break;
                            }
                        }
                    }
                    if (activeCount <= 0)
                        break;
                }
            }
            return results;
        }

    }
}
