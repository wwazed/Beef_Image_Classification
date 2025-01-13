import org.tensorflow.*
import org.tensorflow.keras.*
import org.tensorflow.keras.preprocessing.image.ImageDataGenerator
import org.tensorflow.keras.callbacks.EarlyStopping
import org.tensorflow.keras.callbacks.LearningRateScheduler
import org.tensorflow.keras.layers.*
import org.tensorflow.keras.models.*
import org.tensorflow.keras.optimizers.Adam
import java.io.File
import kotlin.math.cos
import kotlin.math.max
import kotlin.math.min
import kotlin.math.PI

fun main() {
    val trainPath = "/Users/wasifwazed/Documents/Python Scripts/Beef Photo Background Remove/TRAIN"
    val testPath = "/Users/wasifwazed/Documents/Python Scripts/Beef Photo Background Remove/TEST"

    val imgHeight = 224
    val imgWidth = 224
    val batchSize = 32
    val lrRate = 1e-3
    val epochs = 50

    // Data Augmentation and Preprocessing
    val trainDataGen = ImageDataGenerator(
        rescale = 1.0 / 255,
        validationSplit = 0.2
    )
    val trainGenerator = trainDataGen.flowFromDirectory(
        trainPath,
        targetSize = intArrayOf(imgHeight, imgWidth),
        batchSize = batchSize,
        subset = "training"
    )

    val valGenerator = trainDataGen.flowFromDirectory(
        trainPath,
        targetSize = intArrayOf(imgHeight, imgWidth),
        batchSize = batchSize,
        subset = "validation"
    )

    // Model Development
    val baseModel = Applications.ResNet50(
        weights = "imagenet",
        inputShape = intArrayOf(imgHeight, imgWidth, 3),
        includeTop = false
    )
    baseModel.trainable = false

    val inputs = Input(shape = intArrayOf(imgHeight, imgWidth, 3))
    var x = baseModel.apply(inputs)
    x = Conv2D(512, 3, padding = "same", activation = "relu").apply(x)
    x = GlobalAveragePooling2D().apply(x)
    x = Dense(256, activation = "relu").apply(x)
    val outputs = Dense(6).apply(x)

    val model = Model(inputs, outputs)
    model.compile(
        optimizer = Adam(learningRate = lrRate),
        loss = Losses.SparseCategoricalCrossentropy(fromLogits = true),
        metrics = listOf("accuracy")
    )

    // Learning Rate Scheduler
    val lrCallback = LearningRateScheduler { epoch ->
        val lrMax = batchSize * 6e-6
        val lrMin = 1e-5
        val lrRampEp = 3
        if (epoch < lrRampEp) {
            lrMax - lrMin * (epoch.toDouble() / lrRampEp)
        } else {
            lrMax * cos((epoch - lrRampEp) * PI / (epochs - lrRampEp))
        }
    }

    // Training
    val earlyStopping = EarlyStopping(monitor = "val_loss", patience = 5)
    model.fit(
        trainGenerator,
        epochs = epochs,
        validationData = valGenerator,
        callbacks = listOf(earlyStopping, lrCallback)
    )

    // Evaluation
    val testDataGen = ImageDataGenerator(rescale = 1.0 / 255)
    val testGenerator = testDataGen.flowFromDirectory(
        testPath,
        targetSize = intArrayOf(imgHeight, imgWidth),
        batchSize = batchSize
    )
    val results = model.evaluate(testGenerator)
    println("Test Accuracy: ${results[1]}")

    // Save Model
    model.save(File("Beef.keras"))
}
