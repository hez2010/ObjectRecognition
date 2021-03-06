﻿using CNTK;
using OpenCvSharp;
using OpenCvSharp.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace ObjectRecognition
{
    class Program
    {
        static void Main(string[] args)
        {
            //training
            Train();


            //predict:
            Console.WriteLine("Load svm model");
            
            var svm = SVM.Load("SVM_RESULT.xml");
            var cnt = 0;
            Console.WriteLine("Start prediction process");
            var starttime = DateTime.Now;

            //predict videos:

            //foreach (var i in Directory.GetFiles("Videos", "*.mp4"))
            //{
            //    cnt++;
            //    var start = DateTime.Now;
            //    var res = PredictV(svm, i);
            //    Console.WriteLine($"{Path.GetFileName(i)} - Time cost: {(DateTime.Now - start).TotalMilliseconds} ms");
            //    Console.WriteLine("Results:");
            //    for (var j = 1; j < 10; j++)
            //    {
            //        Console.WriteLine($"{j} - confidence: {res.Count(k => k == j) * 100f / res.Count} %");
            //    }
            //}

            //predict imgs
            foreach (var i in Directory.GetFiles("JPEGImages", "*- Copy.jpg")) //test cases
            {
                cnt++;
                var start = DateTime.Now;
                Console.WriteLine($"{Path.GetFileName(i)} - Result: {Predict(svm, Cv2.ImRead(i, ImreadModes.GrayScale))}, time cost: {(DateTime.Now - start).TotalMilliseconds} ms");
            }
            Console.WriteLine($"Average speed: {cnt / (DateTime.Now - starttime).TotalSeconds} per second");

            
            //extract video to images:

            //foreach (var i in Directory.GetFiles("Videos"))
            //{
            //    Console.WriteLine($"Processing: {Path.GetFileName(i)}");
            //    ExportToJpg(i);
            //}

            Console.WriteLine("Finished");
            Console.ReadKey();
            return;
        }

        /// <summary>
        /// predict video
        /// </summary>
        /// <param name="svm"></param>
        /// <param name="fileName"></param>
        /// <returns></returns>
        private static List<float> PredictV(SVM svm, string fileName)
        {
            var res = new List<float>();
            using (VideoCapture videoCapture = new VideoCapture(fileName))
            {
                var total = Convert.ToInt64(videoCapture.Get(CaptureProperty.FrameCount));
                long start = 1;
                videoCapture.Set(CaptureProperty.PosFrames, start);
                long current = start;
                long stop = total;
                Mat frame = new Mat();
                while (current < stop && videoCapture.Read(frame))
                {
                    Cv2.CvtColor(frame, frame, ColorConversionCodes.RGB2GRAY);
                    res.Add(Predict(svm, frame));
                    current++;
                }
                frame.Release();
                videoCapture.Release();
            }
            return res;
        }

        /// <summary>
        /// extract video to images
        /// </summary>
        /// <param name="fileName"></param>
        static void ExportToJpg(string fileName)
        {
            using (VideoCapture videoCapture = new VideoCapture(fileName))
            {
                var total = Convert.ToInt64(videoCapture.Get(CaptureProperty.FrameCount));
                long start = 1;
                videoCapture.Set(CaptureProperty.PosFrames, start);
                long stop = total;
                Mat frame = new Mat();
                long current = start;

                while (current < stop && videoCapture.Read(frame))
                {
                    frame.SaveImage("Videos\\" + Path.GetFileNameWithoutExtension(fileName) + $"_{current}.jpg");
                    current++;
                }
                videoCapture.Release();
            }

        }

        /// <summary>
        /// train model
        /// </summary>
        static void Train()
        {
            //load catalog
            var paras = File.ReadAllLines("JPEGImages\\SVM_Train.txt");
            var imgPath = new List<string>();
            var imgCatg = new List<int>();
            var nImgCnt = paras.Length;
            foreach (var i in paras)
            {
                var temp = i.Split(':');
                imgPath.Add(temp[1]);
                imgCatg.Add(Convert.ToInt32(temp[0]));
            }

            //create HOG
            var hog = new HOGDescriptor(new Size(64, 64), new Size(16, 16), new Size(8, 8), new Size(8, 8));

            Mat data = Mat.Zeros(nImgCnt, hog.GetDescriptorSize(), MatType.CV_32FC1);
            Mat res = Mat.Zeros(nImgCnt, 1, MatType.CV_32SC1);
            for (var z = 0; z < nImgCnt; z++)
            {
                //load img
                Mat src = Cv2.ImRead(imgPath[z], ImreadModes.GrayScale);
                Console.WriteLine($"Processing: {Path.GetFileNameWithoutExtension(imgPath[z])}");

                //resize to 64*64
                Cv2.Resize(src, src, new Size(64, 64));

                //threshold
                src = src.Threshold(200, 255, ThresholdTypes.Binary);

                //center image
                MoveToCenter(src);

                //computer descriptors
                var descriptors = hog.Compute(src, new Size(1, 1), new Size(0, 0));
                for (var i = 0; i < descriptors.Length; i++)
                {
                    data.Set(z, i, descriptors[i]);
                }
                res.Set(z, 0, imgCatg[z]);
                src.Release();
            }

            Console.WriteLine("Start training");

            //create svm
            var svm = SVM.Create();
            svm.TermCriteria = new TermCriteria(CriteriaType.Eps, 1000, float.Epsilon);
            svm.Type = SVM.Types.CSvc;
            svm.KernelType = SVM.KernelTypes.Rbf;
            svm.Degree = 10;
            svm.Gamma = 0.09;
            svm.Coef0 = 1;
            svm.C = 10;
            svm.Nu = 0.5;
            svm.P = 1;

            //training
            svm.Train(data, SampleTypes.RowSample, res);

            //save result
            svm.Save("SVM_RESULT.xml");
        }

        /// <summary>
        /// center image
        /// </summary>
        /// <param name="img"></param>
        static void MoveToCenter(Mat img)
        {
            int left = img.Rows - 1, right = 0, top = img.Rows - 1, buttom = 0;
            for (var i = 0; i < img.Rows; i++)
            {
                for (var j = 0; j < img.Cols; j++)
                {
                    if (img.At<byte>(i, j) != 0)
                    {
                        left = Math.Min(left, j);
                        right = Math.Max(right, j);

                        top = Math.Min(top, i);
                        buttom = Math.Max(buttom, i);
                    }
                }
            }
            int dx = (right + left - img.Rows) / 2, dy = (buttom + top - img.Cols) / 2;
            Mat dst = Mat.Zeros(img.Rows, img.Cols, img.Type());

            for (var i = 0; i < img.Rows; i++)
            {
                for (var j = 0; j < img.Cols; j++)
                {
                    if (i + dx < img.Rows && i + dx >= 0 && j + dy < img.Cols && j + dy >= 0)
                        dst.Set(i + dx, j + dy, img.At<byte>(i, j));
                }
            }
            dst.CopyTo(img);
        }

        /// <summary>
        /// predict
        /// </summary>
        /// <param name="svm"></param>
        /// <param name="src"></param>
        /// <returns></returns>
        static float Predict(SVM svm, Mat src)
        {
            var hog = new HOGDescriptor(new Size(64, 64), new Size(16, 16), new Size(8, 8), new Size(8, 8));
            Mat data = Mat.Zeros(1, hog.GetDescriptorSize(), MatType.CV_32FC1);

            Cv2.Resize(src, src, new Size(64, 64));

            src = src.Threshold(200, 255, ThresholdTypes.Binary);

            MoveToCenter(src);

            var descriptors = hog.Compute(src, new Size(1, 1), new Size(0, 0));
            for (var i = 0; i < descriptors.Length; i++)
            {
                data.Set(0, i, descriptors[i]);
            }
            return svm.Predict(data);
        }
    }
}
