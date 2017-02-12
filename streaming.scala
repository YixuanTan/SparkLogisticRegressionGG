package org.rpi.spark

  /*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// scalastyle:off println

import org.apache.spark.SparkConf
import org.apache.spark.streaming._
import StreamingContext._
import org.apache.log4j.Logger
import org.apache.log4j.Level

/**
 * Counts words in UTF8 encoded, '\n' delimited text received from the network every second.
 *
 * Usa}
// scalastyle:on printlnge: NetworkWordCount <hostname> <port>
 * <hostname> and <port> describe the TCP server that Spark Streaming would connect to receive data.
 *
 * To run this on your local machine, you need to first run a Netcat server
 *    `$ nc -lk 9999`
 * and then run the example
 *    `$ bin/run-example org.apache.spark.examples.streaming.NetworkWordCount localhost 9999`
 */
object streaming {
  def main(args: Array[String]) {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    //StreamingExamples.setStreamingLogLevels()
    val sparkConf = new SparkConf().setAppName("streaming").setMaster("local[2]").set("spark.executor.memory","8g");
    // Create the context
    val ssc = new StreamingContext(sparkConf, Seconds(2))
    //val ssc = new StreamingContext.textFileStream("/home/smartcoder/Documents/Developer/MSthesisData/data2")
    // Create the FileInputDStream on the directory and use the
    // stream to count words in new files created
    val lines = ssc.textFileStream("/home/smartcoder/Documents/Developer/MSthesisData/streaming")
    val words = lines.flatMap(_.split(" "))
    val wordCounts = words.map(x => (x, 1)).reduceByKey(_ + _)
    wordCounts.print()
    ssc.start()
    ssc.awaitTermination()
  }

}