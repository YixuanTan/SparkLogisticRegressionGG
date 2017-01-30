package org.rpi.spark

// Import the needed libraries
import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.mllib.classification.{LogisticRegressionWithLBFGS, LogisticRegressionModel}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.log4j.Logger
import org.apache.log4j.Level


object gglr {
  // Transform each qualitative data in the data set into a double numeric value
  def main(args: Array[String]) {      
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    // Read data into memory in - lazy loading
    val sc = new SparkContext("local[*]","LogisticRegression")
    val data = sc.textFile("/home/smartcoder/Documents/Developer/MSthesisData/*/{[5-9],1[0-5]}_*")
    println(data.count() + "\n")  
    
    // Prepare data for the logistic regression algorithm
    val parsedData = data.map{line => 
        val parts = line.split(",")
        LabeledPoint((parts(5)).toDouble, Vectors.dense(parts.slice(0,5).map(x => x.toDouble)))
    }
    
    //println(parsedData.take(3).mkString("\n"))
    
    // Split data into training (60%) and test (40%)
    val splits = parsedData.randomSplit(Array(0.8, 0.2), seed = 11L)
    val trainingData = splits(0)
    val testData = splits(1)
    
    // Train the model
    val model = new LogisticRegressionWithLBFGS().setNumClasses(2).run(trainingData)
    
    // Evaluate model on training examples and compute training error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    
    val trainErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count
    println("Training Error = " + trainErr)
  
  }

}