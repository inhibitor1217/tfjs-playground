import * as tf from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'

type Car = {
    mpg: number,
    horsepower: number,
}

function getCars() {
    return fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json')
        .then(res => res.json())
        .then((data: Record<string, unknown>[]) => data
            .map(car => ({
                mpg: car.Miles_per_Gallon,
                horsepower: car.Horsepower,
            }))
            .filter(car => (
                car.mpg != null &&
                car.horsepower != null &&
                typeof car.mpg === 'number' &&
                typeof car.horsepower === 'number'
            ))) as Promise<Car[]>
}

function convertToTensor(cars: Car[]) {
    return tf.tidy(() => {
        tf.util.shuffle(cars)

        const inputs = cars.map(c => c.horsepower)
        const labels = cars.map(c => c.mpg)

        const inputTensor = tf.tensor2d(inputs, [inputs.length, 1])
        const labelTensor = tf.tensor2d(labels, [labels.length, 1])

        const inputMax = inputTensor.max()
        const inputMin = inputTensor.min()
        const labelMax = labelTensor.max()
        const labelMin = labelTensor.min()

        const inputsNormalized = inputTensor.sub(inputMin).div(inputMax.sub(inputMin))
        const labelsNormalized = labelTensor.sub(labelMin).div(labelMax.sub(labelMin))

        return {
            inputs: inputsNormalized,
            labels: labelsNormalized,
            bounds: {
                inputMax,
                inputMin,
                labelMax,
                labelMin,
            },
        }
    })
}

function createModel() {
    const model = tf.sequential()

    model.add(tf.layers.dense({ inputShape: [1], units: 8, activation: 'relu', useBias: true }))
    model.add(tf.layers.dense({ units: 8, activation: 'relu', useBias: true }))
    model.add(tf.layers.dense({ units: 1, useBias: true }))

    return model
}

async function trainModel(
    model: tf.LayersModel,
    inputs: tf.Tensor,
    labels: tf.Tensor,
) {
    model.compile({
        optimizer: tf.train.adam(),
        loss: tf.losses.meanSquaredError,
        metrics: ['mse'],
    })

    return await model.fit(inputs, labels, {
        batchSize: 32,
        epochs: 150,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: 'Training Performance' },
            ['loss', 'mse'],
            { height: 200, callbacks: ['onEpochEnd'] },
        )
    })
}

function testModel(
    model: tf.LayersModel,
    input: Car[],
    bounds: {
        inputMin: tf.Tensor,
        inputMax: tf.Tensor,
        labelMin: tf.Tensor,
        labelMax: tf.Tensor,
    },
) {
    const { inputMin, inputMax, labelMin, labelMax } = bounds

    const [x, pred] = tf.tidy(() => {
        const xNorm = tf.linspace(0, 1, 100)
        const predictions = model.predict(xNorm.reshape([100, 1])) as tf.Tensor

        const xUnnorm = xNorm.mul(inputMax.sub(inputMin)).add(inputMin)
        const predUnnorm = predictions.mul(labelMax.sub(labelMin)).add(labelMin)

        return [xUnnorm.dataSync(), predUnnorm.dataSync()]
    })

    const predictedPoints = Array.from(x)
        .map((x, i) => ({ x, y: pred[i] }))

    const originalPoints = input.map(c => ({ x: c.horsepower, y: c.mpg }))

    tfvis.render.scatterplot(
        { name: 'Model predictions vs Original Data' },
        { values: [originalPoints, predictedPoints], series: ['original', 'predicted'] },
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300
        },
    )
}

async function run() {
    const cars = await getCars()

    const values = cars.map(car => ({ x: car.horsepower, y: car.mpg }))
    tfvis.render.scatterplot(
        { name: 'Horsepower v MPG' },
        { values },
        {
            xLabel: 'Horsepower',
            yLabel: 'MPG',
            height: 300,
        }
    )

    const model = createModel()
    tfvis.show.modelSummary({ name: 'Model Summary' }, model)

    const { inputs, labels, bounds } = convertToTensor(cars)
    await trainModel(model, inputs, labels)
    console.log('Done Training')

    testModel(model, cars, bounds)
}

document.addEventListener('DOMContentLoaded', run)
