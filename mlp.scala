package org.rpi.spark

// Import the needed libraries
import org.apache.log4j.Logger
import org.apache.log4j.Level
import sys.process._
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator


object mlp {
  // Transform each qualitative data in the data set into a double numeric value
  def main(args: Array[String]) {      
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)
    "rm -r /home/smartcoder/Documents/Developer/MSthesisData/ROC/"!
    // Read data into memory in - lazy loading
    //val sc = new SparkContext("local[*]","LogisticRegression")
    //val data = sc.textFile("/home/smartcoder/Documents/Developer/MSthesisData/*/{[5-9],1[0-5]}_[6-7]*_[2-4]*")
    //val data = sc.textFile("/home/smartcoder/Documents/Developer/MSthesisData/data1/{[5-9],1[0-5]}_[6-7]*_[2-4]*")
    import org.apache.spark.sql.SparkSession
    val spark = SparkSession.builder
      .master("local[*]")
      .appName("multi-perceptron")
      .config("spark.some.config.option", "config-value")
      .getOrCreate()
    val data = spark.read.format("libsvm").load("/home/smartcoder/Documents/Developer/MSthesisData/libsvm_formatted.txt")

    val count = data.count()
    println(data.count() + "\n") 
       
    // Split the data into train and test
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
    val train = splits(0)
    val test = splits(1)

    
    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val layers = Array[Int](5, 20, 20, 2)
    
    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(10000)
    
    // train the model
    val model = trainer.fit(train)
    
    // compute accuracy on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("accuracy")

    
    println("Test set accuracy = " + evaluator.evaluate(predictionAndLabels))    
    
  }

}