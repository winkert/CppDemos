using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using TRW.CommonLibraries.NeuralNetwork;
using TRW.UnitTesting;

namespace TRW.CommonLibraries.NeuralNetwork.Test
{
    [TestClass]
    public class OnnxerTests: UnitTestBase
    {
        [TestMethod]
        public void TestSaveOnnxFile()
        {

        }

        [TestMethod]
        public void TestLoadOnnxFile() 
        {
            string onnxFile = UnitTestBase.MoveFileToCurrentWorkingDirectory("bert_Opset18.onnx");
            Onnxer onnxer = new Onnxer();
            TRW.CommonLibraries.NeuralNetwork.NeuralNetwork target = onnxer.ImportFromOnnx(onnxFile);
            Assert.IsNotNull(target);
        }
    }
}
