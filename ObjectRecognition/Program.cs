using CNTK;
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
            //predict:
            Console.WriteLine("Load svm model");
            var svm = SVM.Load("SVM_RESULT.xml");
            var cnt = 0;
            Console.WriteLine("Start prediction process");
            var starttime = DateTime.Now;
            foreach (var i in Directory.GetFiles("Videos", "*.mp4"))
            {
                cnt++;
                var start = DateTime.Now;
                var res = PredictV(svm, i);
                Console.WriteLine($"{Path.GetFileName(i)} - Time cost: {(DateTime.Now - start).TotalMilliseconds} ms");
                Console.WriteLine("Results:");
                for(var j = 1; j < 10; j++)
                {
                    Console.WriteLine($"{j} - confidence: {res.Count(k => k == j) * 100f / res.Count} %");
                }
            }
            foreach (var i in Directory.GetFiles("JPEGImages", "*.jpg"))
            {
                cnt++;
                var start = DateTime.Now;
                Console.WriteLine($"{Path.GetFileName(i)} - Result: {Predict(svm, i)}, time cost: {(DateTime.Now - start).TotalMilliseconds} ms");
            }
            Console.WriteLine($"Average speed: {cnt / (DateTime.Now - starttime).TotalSeconds} per second");



            //train svm model:
            //Train();



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
                    res.Add(Predict(svm, frame));
                    current++;
                }
                videoCapture.Release();
            }
            return res;
        }

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

            //creat matrix
            Mat data = Mat.Zeros(nImgCnt, 144, MatType.CV_32FC1);
            Mat res = Mat.Zeros(nImgCnt, 1, MatType.CV_32SC1);
            for (var z = 0; z < nImgCnt; z++)
            {
                //load img
                Mat src = Cv2.ImRead(imgPath[z]);
                Console.WriteLine($"Processing: {Path.GetFileNameWithoutExtension(imgPath[z])}");
                Cv2.Resize(src, src, new Size(64, 64));
                var hog = new HOGDescriptor(new Size(64, 64), new Size(16, 16), new Size(16, 16), new Size(16, 16), 9);
                var descriptors = hog.Compute(src, new Size(8, 8), new Size(0, 0));
                for (var i = 0; i < descriptors.Length; i++)
                {
                    data.Set(z, i, descriptors[i]);
                }
                res.Set(z, 0, imgCatg[z]);
            }

            Console.WriteLine("Start training");

            //create svm
            var svm = SVM.Create();
            svm.TermCriteria = new TermCriteria(CriteriaType.Eps, 1000, float.Epsilon);
            svm.Type = SVM.Types.CSvc;
            svm.KernelType = SVM.KernelTypes.Rbf;
            svm.Degree = 10;
            svm.Gamma = 8;
            svm.Coef0 = 1;
            svm.C = 10;
            svm.Nu = 0.5;
            svm.P = 0.1;

            //training
            svm.Train(data, SampleTypes.RowSample, res);

            //save result
            svm.Save("SVM_RESULT.xml");
        }

        static float Predict(SVM svm, Mat src)
        {
            Mat data = Mat.Zeros(1, 144, MatType.CV_32FC1);
            Cv2.Resize(src, src, new Size(64, 64));

            var hog = new HOGDescriptor(new Size(64, 64), new Size(16, 16), new Size(16, 16), new Size(16, 16), 9);
            var descriptors = hog.Compute(src, new Size(8, 8), new Size(0, 0));
            for (var i = 0; i < descriptors.Length; i++)
            {
                data.Set(0, i, descriptors[i]);
            }

            return svm.Predict(data);
        }

        static float Predict(SVM svm, string fileName)
        {
            var src = Cv2.ImRead(fileName);

            Mat data = Mat.Zeros(1, 144, MatType.CV_32FC1);
            Cv2.Resize(src, src, new Size(64, 64));

            var hog = new HOGDescriptor(new Size(64, 64), new Size(16, 16), new Size(16, 16), new Size(16, 16), 9);
            var descriptors = hog.Compute(src, new Size(8, 8), new Size(0, 0));
            for (var i = 0; i < descriptors.Length; i++)
            {
                data.Set(0, i, descriptors[i]);
            }

            return svm.Predict(data);
        }
    }
}
