/**
 * Multi-Layer Perceptron (MLP) for NLU
 *
 * Single hidden-layer neural network with:
 * - Xavier weight initialization
 * - ReLU activation on hidden layer
 * - Softmax output with cross-entropy loss
 * - SGD with L2 regularization and learning rate decay
 * - Full-batch training (small NLU datasets)
 *
 * Architecture: inputDim → hiddenDim → numClasses
 * Default: 1024 → 128 → 12 = 132,748 parameters
 *
 * Inference: O(inputDim * hiddenDim + hiddenDim * numClasses)
 * Two matrix-vector multiplies + ReLU + softmax = microseconds
 */

export interface MLPModel {
  type: "mlp";
  W1: Float32Array; // inputDim × hiddenDim (row-major)
  b1: Float32Array; // hiddenDim
  W2: Float32Array; // hiddenDim × numClasses (row-major)
  b2: Float32Array; // numClasses
  inputDim: number;
  hiddenDim: number;
  classes: string[];
}

/**
 * Xavier initialization: N(0, sqrt(2 / (fan_in + fan_out)))
 */
function xavierInit(fanIn: number, fanOut: number, size: number): Float32Array {
  const std = Math.sqrt(2.0 / (fanIn + fanOut));
  const arr = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    // Box-Muller transform for Gaussian
    const u1 = Math.random() || 1e-10;
    const u2 = Math.random();
    arr[i] = std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }
  return arr;
}

/**
 * Train MLP with SGD + backpropagation
 */
export function trainMLP(
  vectors: Float32Array[],
  labels: string[],
  inputDim: number,
  options: {
    hiddenDim?: number;
    epochs?: number;
    learningRate?: number;
    lambda?: number;
  } = {},
): MLPModel {
  const {
    hiddenDim = 128,
    epochs = 50,
    learningRate = 0.01,
    lambda = 0.0001,
  } = options;

  const classes = [...new Set(labels)];
  const numClasses = classes.length;
  const classIdx: Record<string, number> = {};
  for (let c = 0; c < numClasses; c++) classIdx[classes[c]] = c;

  const n = vectors.length;

  // Initialize weights
  const W1 = xavierInit(inputDim, hiddenDim, inputDim * hiddenDim);
  const b1 = new Float32Array(hiddenDim);
  const W2 = xavierInit(hiddenDim, numClasses, hiddenDim * numClasses);
  const b2 = new Float32Array(numClasses);

  // Gradient accumulators
  const dW1 = new Float32Array(inputDim * hiddenDim);
  const db1 = new Float32Array(hiddenDim);
  const dW2 = new Float32Array(hiddenDim * numClasses);
  const db2 = new Float32Array(numClasses);

  // Buffers for forward pass
  const hidden = new Float32Array(hiddenDim);
  const output = new Float32Array(numClasses);
  const dHidden = new Float32Array(hiddenDim);

  for (let epoch = 0; epoch < epochs; epoch++) {
    const lr = learningRate * (1 - epoch / epochs); // linear decay

    // Zero gradients
    dW1.fill(0);
    db1.fill(0);
    dW2.fill(0);
    db2.fill(0);

    for (let s = 0; s < n; s++) {
      const x = vectors[s];
      const target = classIdx[labels[s]];

      // --- Forward pass ---
      // Hidden: h = ReLU(W1 · x + b1)
      for (let j = 0; j < hiddenDim; j++) {
        let sum = b1[j];
        const offset = j * inputDim;
        for (let i = 0; i < inputDim; i++) {
          sum += W1[offset + i] * x[i];
        }
        hidden[j] = sum > 0 ? sum : 0; // ReLU
      }

      // Output: z = W2 · h + b2, then softmax
      let maxZ = -Infinity;
      for (let k = 0; k < numClasses; k++) {
        let sum = b2[k];
        const offset = k * hiddenDim;
        for (let j = 0; j < hiddenDim; j++) {
          sum += W2[offset + j] * hidden[j];
        }
        output[k] = sum;
        if (sum > maxZ) maxZ = sum;
      }

      let expSum = 0;
      for (let k = 0; k < numClasses; k++) {
        output[k] = Math.exp(output[k] - maxZ);
        expSum += output[k];
      }
      for (let k = 0; k < numClasses; k++) {
        output[k] /= expSum;
      }

      // --- Backward pass ---
      // dL/dz = softmax_output - one_hot(target)
      // (softmax + cross-entropy gradient)
      for (let k = 0; k < numClasses; k++) {
        output[k] -= (k === target ? 1 : 0);
      }

      // dW2 += outer(dz, hidden), db2 += dz
      for (let k = 0; k < numClasses; k++) {
        const dzk = output[k];
        db2[k] += dzk;
        const offset = k * hiddenDim;
        for (let j = 0; j < hiddenDim; j++) {
          dW2[offset + j] += dzk * hidden[j];
        }
      }

      // dHidden = W2^T · dz, masked by ReLU derivative
      for (let j = 0; j < hiddenDim; j++) {
        let sum = 0;
        for (let k = 0; k < numClasses; k++) {
          sum += W2[k * hiddenDim + j] * output[k];
        }
        dHidden[j] = hidden[j] > 0 ? sum : 0; // ReLU gradient
      }

      // dW1 += outer(dHidden, x), db1 += dHidden
      for (let j = 0; j < hiddenDim; j++) {
        const dhj = dHidden[j];
        db1[j] += dhj;
        const offset = j * inputDim;
        for (let i = 0; i < inputDim; i++) {
          dW1[offset + i] += dhj * x[i];
        }
      }
    }

    // --- SGD update with L2 regularization ---
    const scale = lr / n;
    const decay = 1 - lr * lambda;

    for (let i = 0; i < W1.length; i++) {
      W1[i] = W1[i] * decay - dW1[i] * scale;
    }
    for (let j = 0; j < hiddenDim; j++) {
      b1[j] -= db1[j] * scale;
    }
    for (let i = 0; i < W2.length; i++) {
      W2[i] = W2[i] * decay - dW2[i] * scale;
    }
    for (let k = 0; k < numClasses; k++) {
      b2[k] -= db2[k] * scale;
    }
  }

  return { type: "mlp", W1, b1, W2, b2, inputDim, hiddenDim, classes };
}

/**
 * Predict with MLP — two matrix-vector multiplies + softmax
 */
export function predictMLP(
  vector: Float32Array,
  model: MLPModel,
): Array<{ intent: string; score: number }> {
  const { W1, b1, W2, b2, inputDim, hiddenDim, classes } = model;
  const numClasses = classes.length;

  // Hidden: h = ReLU(W1 · x + b1)
  const hidden = new Float32Array(hiddenDim);
  for (let j = 0; j < hiddenDim; j++) {
    let sum = b1[j];
    const offset = j * inputDim;
    for (let i = 0; i < inputDim; i++) {
      sum += W1[offset + i] * vector[i];
    }
    hidden[j] = sum > 0 ? sum : 0;
  }

  // Output: z = W2 · h + b2
  const scores: Array<{ intent: string; score: number }> = [];
  let maxZ = -Infinity;
  const raw = new Float32Array(numClasses);
  for (let k = 0; k < numClasses; k++) {
    let sum = b2[k];
    const offset = k * hiddenDim;
    for (let j = 0; j < hiddenDim; j++) {
      sum += W2[offset + j] * hidden[j];
    }
    raw[k] = sum;
    if (sum > maxZ) maxZ = sum;
  }

  // Softmax
  let expSum = 0;
  for (let k = 0; k < numClasses; k++) {
    raw[k] = Math.exp(raw[k] - maxZ);
    expSum += raw[k];
  }
  for (let k = 0; k < numClasses; k++) {
    scores.push({ intent: classes[k], score: raw[k] / expSum });
  }

  scores.sort((a, b) => b.score - a.score);
  return scores;
}

/**
 * Serialize MLP — Float32Array → number[] for JSON
 */
export function serializeMLP(model: MLPModel): object {
  return {
    ...model,
    W1: Array.from(model.W1),
    b1: Array.from(model.b1),
    W2: Array.from(model.W2),
    b2: Array.from(model.b2),
  };
}

/**
 * Deserialize MLP — number[] → Float32Array
 */
export function deserializeMLP(data: Record<string, unknown>): MLPModel {
  const d = data as Record<string, unknown>;
  return {
    type: "mlp",
    W1: new Float32Array(d.W1 as number[]),
    b1: new Float32Array(d.b1 as number[]),
    W2: new Float32Array(d.W2 as number[]),
    b2: new Float32Array(d.b2 as number[]),
    inputDim: d.inputDim as number,
    hiddenDim: d.hiddenDim as number,
    classes: d.classes as string[],
  };
}
