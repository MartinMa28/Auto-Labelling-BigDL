package com.intel.analytics.bigdl.models.resnet
import java.awt.image.{BufferedImage, DataBufferByte}

import org.apache.hadoop.hbase.util.Bytes
import HBaseHelperAPI.HBaseHelperAPI._
import com.intel.analytics.bigdl.nn.{LogSoftMax, Module, SoftMax}
import com.intel.analytics.bigdl.utils.{Engine, T}
import com.intel.analytics.bigdl.tensor.Tensor
import com.intel.analytics.bigdl.transform.vision.image._
import javax.imageio.ImageIO
import org.apache.spark.SparkContext
import com.intel.analytics.bigdl.numeric.NumericFloat
import com.intel.analytics.bigdl.transform.vision.image.augmentation.{Resize, _}
import java.io.{ByteArrayInputStream, File}

import org.apache.hadoop.hbase.client._

import scala.collection.JavaConversions._
import java.net._
import java.io._

import com.intel.analytics.bigdl.optim.Top1Accuracy


object test {

  def main(args: Array[String]): Unit = {
    if (args.length == 0) {
      println("arguments: core-site path, hbase-site path, model_path, table name")
    }else {
      for (arg <- args){
        println(arg)
      }
    }
    // create the socket server object
    val server = new ServerSocket(10001)
    val socketFinishLoading = server.accept()

    val conf = Engine.createSparkConf().setAppName("Test ResNet")
      .set("spark.akka.frameSize", 64.toString)
      .set("spark.task.maxFailures", "1")
    val sc = new SparkContext(conf)

    Engine.init
    println(Engine.getEngineType())

    println("===========Load module and connect to HBase table====================")
    val model = Module.loadModule(args(2))
    //val model = Module.load(args(2))
    val table = connectToHBase(args(0), args(1), args(3))
    println("============Finish loading===============")


    val outFinishLoading = new PrintWriter(socketFinishLoading.getOutputStream, true)
    outFinishLoading.println("Finish loading")
    socketFinishLoading.close()

    while(true){
      println("Waiting for arguments: start row, stop row, number of rows")
      val socket = server.accept()
      // socket reader
      val in = new BufferedReader(new InputStreamReader(socket.getInputStream))
      // socket writer
      val out = new PrintWriter(socket.getOutputStream, true)

      val msgStr = in.readLine()
      val msgArgs = msgStr.split(" ")
      for (arg <- msgArgs){
        println(arg)
      }
      val toClient = "%d".format(msgArgs.length)
      //out.println(toClient)

      // if the client sends "stop" message, then exits constant while loop and stop the spark context
      if (msgStr == "stop"){
        server.close()
        sc.stop()
        return
      }

      val startRow = Bytes.toBytes(msgArgs(0))
      val stopRow = Bytes.toBytes(msgArgs(1))
      val family = Bytes.toBytes("123_s20")
      val qualifiers = Array(Bytes.toBytes("data"))
      val numOfImages = msgArgs(2).toInt


      val retrievedStrings = scanGetData(table, startRow, stopRow, numOfImages, family, qualifiers)
      val numOfPredictions = retrievedStrings.length
      println("predictions: " + numOfPredictions)

      val valRdd = sc.parallelize(retrievedStrings)

      val row, col = 4
      val eWidth, eHeight = 150
      val step = 120
      val numOfWindows = row * col

      val validateSet = valRdd.flatMap(rowString => {
        val rowKey: String = rowString.split(";")(0)

        val base64String: String = rowString.split(";")(1)
        val rawBytes: Array[Byte] = javax.xml.bind.DatatypeConverter
          .parseBase64Binary(base64String.map { case '-' => '+'; case '_' => '/'; case c => c })

        val bis = new ByteArrayInputStream(rawBytes)
        val image = ImageIO.read(bis)
        bis.close()

        val imf : Array[ImageFeature] = new Array[ImageFeature](numOfWindows)
        for(i <- 0 until numOfWindows){
          imf(i) = new ImageFeature()
        }

        var y, x = 0
        for(i <- 0 until row){
          x = 0
          for(j <- 0 until col){
            val subImage = new BufferedImage(eWidth, eHeight, 5)
            val g = subImage.getGraphics
            g.drawImage(image.getSubimage(x, y, eWidth, eHeight),0,0,null)
            g.dispose()
            val bytes: Array[Byte] = subImage.getRaster.getDataBuffer.asInstanceOf[DataBufferByte].getData()
            val index = i*row + j

            //          val fileName = "/home/yilinma/Pictures/cropped_images/" + rowKey + "_" + index + ".jpg"
            //          ImageIO.write(subImage, "jpg", new File(fileName))


            // ImageFeature refers to the object "ImageFeature", which has some string fields
            // and then, call the imf(index)'s apply function
            imf(index)("bytes") = bytes
            imf(index)("originalSize") = (eWidth, eHeight, 3)
            imf(index)("rowKey") = rowKey
            imf(index)("index") = index
            imf(index)("x") = j * step
            imf(index)("y") = i * step
            x += step
          }
          y += step
        }
        imf
      })

      println("===========Print rdd====================")
      val repartitionedValidateSet = validateSet.repartition(1)
      println("Partitions:" + repartitionedValidateSet.partitions.size)
      // distributed image frame, which is going to be sent to the model
      val distributedImageFrame = ImageFrame.rdd(repartitionedValidateSet) ->
        PixelBytesToMat()->
        Resize(256, 256) ->
        CenterCrop(224, 224) ->
        ChannelScaledNormalizer(104, 117, 123, 0.0078125) ->
        MatToTensor() -> ImageFrameToSample()

      println("===========transform over====================")
      println("===========Predice Image====================")
      val result = model.predictImage(distributedImageFrame).toDistributed()

      println("====================select keys================")
      val keys = result.rdd.map(r => {
        val rowKey = r("rowKey").asInstanceOf[String]
        val x = r("x").asInstanceOf[Int]
        val y = r("y").asInstanceOf[Int]
        val pred = r("predict").asInstanceOf[Tensor[Float]]
        val index = r("index").asInstanceOf[Int]
        Map("rowKey" -> rowKey, "index" -> index, "x" -> x, "y" -> y, "predict" -> pred)
      })
      println("=================Print Result=======================")
      val features = keys.collect()


      for (f <- features){
        println(f("rowKey"))
        println(f("index"))
        println(f("x").toString + "," + f("y").toString)
        println(f("predict"))
        val key: String = f("rowKey").asInstanceOf[String]
        val ind: String = f("index").toString

        val resultTensor = f("predict").asInstanceOf[Tensor[Float]]
        val resultArray = resultTensor.toArray()

        if (resultArray(0) < resultArray(1)){
          println("positive")
        }else {
          println("negative")
        }
        // positive: booleanLabel == True, negative: booleanLabel == False
        //      val booleanLabel = resultTensor.valueAt(1) > resultTensor.valueAt(0)
        //      val intLabel = if (booleanLabel) 1 else 0
        //      println("Label: ", intLabel)
        //      println("tensor: " + resultTensor)
        //      val softMaxLayer = SoftMax()
        //      val output = softMaxLayer.forward(resultTensor)
        //      println("softmax: " + output)

      }

      println("==================Start Putting===================")
      var putList = new java.util.ArrayList[Put]()
      for (i <- 0 until numOfPredictions){
        var label = 0
        var logitMax = 0.0
        var xCoordinate = 0
        var yCoordinate = 0
        val curRowKey: String = features(i * numOfWindows)("rowKey").asInstanceOf[String]
        for (j <- 0 until numOfWindows){
          val feature = features(i * numOfWindows + j)
          val resultArray = feature("predict").asInstanceOf[Tensor[Float]].toArray()
          if (resultArray(0) < resultArray(1)){
            // positive
            label += 1
            if (resultArray(1) > logitMax){
              logitMax = resultArray(1)
              xCoordinate = feature("x").asInstanceOf[Int]
              yCoordinate = feature("y").asInstanceOf[Int]
            }

          }
        }
        val putObj = new Put(Bytes.toBytes(curRowKey))
        if (label > 0){
          // if label > 0, put (xCoordinate, yCoordinate), pos 1
          putObj.addColumn(Bytes.toBytes("123_s20"), Bytes.toBytes("pos"), Bytes.toBytes("1"))
          putObj.addColumn(Bytes.toBytes("123_s20"), Bytes.toBytes("offset"), Bytes.toBytes(xCoordinate.toString + "," + yCoordinate))
        }
        else{
          // else put # pos 0
          putObj.addColumn(Bytes.toBytes("123_s20"), Bytes.toBytes("pos"), Bytes.toBytes("0"))
          putObj.addColumn(Bytes.toBytes("123_s20"), Bytes.toBytes("offset"), Bytes.toBytes("#"))
        }

        putList += putObj
      }
      table.put(putList)
      table.close()
      println("==================Stop putting===================")
      // go back to Django process
      out.println(toClient)
      socket.close()
    }

  }

}
