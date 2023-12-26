import * as tf from '@tensorflow/tfjs'

async function run() {
    const model = tf.sequential()
    model.add(tf.layers.dense({ units: 1, inputShape: [1] }))

    model.compile({ loss: 'meanSquaredError', optimizer: 'sgd' })

    const x = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1])
    const y = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1])

    await model.fit(x, y, { epochs: 250 })

    const el = document.getElementById('micro-out-div')
    if (el) {
        el.innerText = model.predict(tf.tensor2d([20], [1, 1])).toString()
    }
}

run()
