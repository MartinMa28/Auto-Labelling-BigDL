package HBaseConnect

import java.awt.image.BufferedImage

import org.apache.hadoop.fs.Path
import org.apache.hadoop.hbase.client.{HBaseAdmin, Result}
import org.apache.hadoop.hbase.io.ImmutableBytesWritable
import org.apache.hadoop.hbase.mapreduce.TableInputFormat
import org.apache.hadoop.hbase.util.Bytes
import org.apache.hadoop.hbase.{HBaseConfiguration, HTableDescriptor}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}


object HBaseConnector {
  case class KFBioRow(rowkey: String, data: BufferedImage, offset: String, pos: String, train: String)

  object KFBioRow extends Serializable {
    def parseKFBioRow(result: Result, columnFamily: String): String = {
      val rowKey = Bytes.toString(result.getRow)
      val columnFamilyBytes = Bytes.toBytes(columnFamily)
      val imageStringData = Bytes.toString(result.getValue(columnFamilyBytes, Bytes.toBytes("data")))

      val offset = Bytes.toString(result.getValue(columnFamilyBytes, Bytes.toBytes("offset")))
      val pos = Bytes.toString(result.getValue(columnFamilyBytes, Bytes.toBytes("pos")))
      val train = Bytes.toString(result.getValue(columnFamilyBytes, Bytes.toBytes("train")))

      rowKey + ";" + offset + ";" + train + ";" + pos + ";" + imageStringData
    }
  }

  def connectToHBase(sc: SparkContext, coreSitePath: String, hbaseSitePath: String, hbaseTalbeName: String,rowKeyStart: String, rowKeyEnd: String): (RDD[(ImmutableBytesWritable, Result)]) = {
    val hbaseConf = HBaseConfiguration.create()

    hbaseConf.addResource(new Path(coreSitePath))
    hbaseConf.addResource(new Path(hbaseSitePath))
    hbaseConf.set(TableInputFormat.SCAN_ROW_START, rowKeyStart)
    hbaseConf.set(TableInputFormat.SCAN_ROW_STOP, rowKeyEnd)
    hbaseConf.set(TableInputFormat.INPUT_TABLE, hbaseTalbeName)

    val admin = new HBaseAdmin(hbaseConf)
    if (!admin.isTableAvailable(hbaseTalbeName)) {
      val tableDesc = new HTableDescriptor(hbaseTalbeName)
      admin.createTable(tableDesc)
    }
    val hbaseRDD = sc.newAPIHadoopRDD(hbaseConf, classOf[TableInputFormat], classOf[ImmutableBytesWritable], classOf[Result])
    hbaseRDD
  }

}
