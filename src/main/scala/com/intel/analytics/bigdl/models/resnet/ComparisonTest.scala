package com.intel.analytics.bigdl.models.resnet
import com.intel.analytics.bigdl.nn.abstractnn.{AbstractModule, Activity}
import com.intel.analytics.bigdl.nn.{Graph, Module, StaticGraph}
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{CenterCrop, ChannelScaledNormalizer, Resize}
import com.intel.analytics.bigdl.transform.vision.image.{BytesToMat, ImageFrame, ImageFrameToSample, MatToTensor}
import com.intel.analytics.bigdl.utils.Engine
import com.intel.analytics.bigdl.utils.intermediate.IRGraph
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.dataset.Sample
import com.intel.analytics.bigdl.optim.Top1Accuracy
import com.intel.analytics.bigdl.models.resnet.test

object ComparisonTest {

  def main(args: Array[String]): Unit = {
    val conf = Engine.createSparkConf().setAppName("Comparison Test for ResNet50")
      .set("spark.akka.frameSize", 64.toString)
      .set("spark.task.maxFailures", "1")

    val sc = new SparkContext(conf)
    Engine.init
    println(Engine.getEngineType())

//    val distributedImageFrame = ImageFrame.read("/home/yilinma/Documents/tmp/imgs", sc) ->
//      BytesToMat() ->
//      Resize(256, 256) ->
//      CenterCrop(224, 224) ->
//      ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
//      MatToTensor[Float]() -> ImageFrameToSample()
    println("before loading")
    var model = Module.loadModule(args(0))
    model = test.modelProcessing(model)
    println("after loading")
    val dummyData = Tensor(3, 224, 224).rand()
    val dummyLabel = Tensor(1)
    val dummySample = Sample(dummyData, dummyLabel)
    val dummySamples = new Array[Sample[Float]](640)
    for (i <- 0 until 640) {
      dummySamples(i) = dummySample
    }
    val dummyRDD = sc.parallelize(dummySamples, args(1).toInt)
    println("dummy RDD partitions: " + dummyRDD.partitions.size)
    val evaluateResult = model.evaluate(dummyRDD, Array(new Top1Accuracy))
    evaluateResult.foreach(r => println(s"${r._2} is ${r._1}"))
//    val result = model.predictImage(distributedImageFrame, batchPerPartition = 4).toDistributed()
//    println("before collect")
//    val features = result.rdd.collect()
//    println("after collect")
//    for (f <- features) {
//      println(f("predict"))
//    }
    sc.stop()
  }
}
