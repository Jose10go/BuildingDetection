using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Linq;
using System.Net;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;

namespace CNTKUtil
{
    /// <summary>
    /// A collection of utilities to work with data files.
    /// </summary>
    public static class DataUtil
    {
        /// <summary>
        /// Get an image reader to sequentially read images from disk for training.
        /// </summary>
        /// <param name="mapFilePath">The path to the map file</param>
        /// <param name="imageWidth">The width to scale all images to</param>
        /// <param name="imageHeight">The height to scale all images to</param>
        /// <param name="numChannels">The number of channels to transform all images to</param>
        /// <param name="numClasses">The number of label classes in this training set</param>
        /// <param name="randomizeData">Set to true to randomize the data for training</param>
        /// <param name="augmentData">Set to true to use data augmentation to expand the training set</param>
        /// <returns>An image source ready for use in training or testing.</returns>
        public static CNTK.MinibatchSource GetImageReader(string mapFilePath, int imageWidth, int imageHeight, int numChannels, int numClasses, bool randomizeData, bool augmentData)
        {
            var transforms = new List<CNTK.CNTKDictionary>();
            if (augmentData)
            {
                var randomSideTransform = CNTK.CNTKLib.ReaderCrop("RandomSide",
                  new Tuple<int, int>(0, 0),
                  new Tuple<float, float>(0.8f, 1.0f),
                  new Tuple<float, float>(0.0f, 0.0f),
                  new Tuple<float, float>(1.0f, 1.0f),
                  "uniRatio");
                transforms.Add(randomSideTransform);
            }
            var scaleTransform = CNTK.CNTKLib.ReaderScale(imageWidth, imageHeight, numChannels);
            transforms.Add(scaleTransform);

            var imageDeserializer = CNTK.CNTKLib.ImageDeserializer(mapFilePath, "labels", (uint)numClasses, "features", transforms);
            var minibatchSourceConfig = new CNTK.MinibatchSourceConfig(new CNTK.DictionaryVector() { imageDeserializer });
            if (!randomizeData)
            {
                minibatchSourceConfig.randomizationWindowInChunks = 0;
                minibatchSourceConfig.randomizationWindowInSamples = 0;
            }
            return CNTK.CNTKLib.CreateCompositeMinibatchSource(minibatchSourceConfig);
        }

    }
}
