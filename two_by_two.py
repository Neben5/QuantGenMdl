import numpy as np
import torch
import argparse
import time
import os
import matplotlib.pyplot as plt
from matplotlib import rc
from src.QDDPM_torch import DiffusionModel, QDDPM, naturalDistance
import src.ImageEncode as ie

rc("text", usetex=False)
rc("axes", linewidth=3)

parser = argparse.ArgumentParser(
    description="""Runs a quantum diffusion algorithm to create a model
    that maps inputs to outputs resembling four black pixels surrounded
    by a circle of white"""
)
parser.add_argument("-g", help="Try to use GPU", action="store_true")
parser.add_argument("dir", help="Target dir for output files")


n, na = 4, 1  # number of data and ancilla qubits
T = 20  # number of diffusion steps
L = 6  # layer of backward PQC
Ndata = 1000  # number of data in the training data set
epochs = Ndata * T  # number of training epochs
dir = ""


def generate_training(
    values: np.array, n_train: int, scale: float, seed=None, debug=False
):
    np.random.seed(seed)
    n = values.size
    noise = abs(np.random.randn(n_train, n)) + 0j * np.random.randn(n_train, n)
    if debug:
        print("Shape of values: %s" % (values.shape,))
        print("Shape of noise: %s" % (noise.shape,))
    states = (noise * scale) + values
    if debug:
        print("Shape of inputs with added and scaled noise: %s" % (states.shape,))
    states /= np.tile(np.linalg.norm(states, axis=1).reshape(1, n_train), (n, 1)).T
    return states


def generate_source():
    temp_test = np.array([[1, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]])
    temp_test = temp_test.flatten()
    return temp_test


def generate_diffusion_data(
    source_values: np.ndarray,
) -> tuple[np.ndarray, DiffusionModel]:
    """Generates diffusion data and model

    Args:
        source_values (np.ndarray): Training data for first diffusion step of dim (Ndata, 2**n)

    Returns:
        np.ndarray: Dimension (T+1, Ndata, x) diffusion states through T + 1
        DiffusionModel: Associated diffusion model
    """
    diff_hs = np.linspace(0.5, 4.0, T)

    model_diff = DiffusionModel(n, T, Ndata)
    X = torch.from_numpy(source_values)
    Xout = np.zeros((T + 1, Ndata, 2**n), dtype=np.complex128)
    Xout[0] = X
    for t in range(1, T + 1):
        Xout[t] = model_diff.set_diffusionData_t(t, X, diff_hs[:t], seed=t).numpy()
        print("  Diffusion step %d" % t)

    np.save("%s/states_diff.npy" % (dir), Xout)
    return Xout, model_diff


def Training_t(model, t, inputs_T, params_tot, Ndata, epochs):
    """
    the training for the backward PQC at step t
    input_t_plus_1: the output from step t+1, as the role of input at step t
    Args:
    model: the QDDPM model
    t: the diffusion step
    inputs_T: the input data at step t=T
    params_tot: collection of PQC parameters before step t
    Ndata: number of samples in dataset
    epochs: the number of iterations
    """
    input_t_plus_1 = model.prepareInput_t(
        inputs_T, params_tot, t, Ndata
    )  # prepare input
    states_diff = model.states_diff
    loss_hist = []  # record of training history

    # initialize parameters
    np.random.seed()
    params_t = torch.tensor(
        np.random.normal(size=2 * model.n_tot * model.L), requires_grad=True
    )
    # set optimizer and learning rate decay
    optimizer = torch.optim.Adam([params_t], lr=0.0005)

    t0 = time.time()
    for step in range(epochs):
        indices = np.random.choice(states_diff.shape[1], size=Ndata, replace=False)
        true_data = states_diff[t, indices]

        output_t = model.backwardOutput_t(input_t_plus_1, params_t)
        loss = naturalDistance(output_t, true_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_hist.append(loss)  # record the current loss

        if step % 100 == 0:
            loss_value = loss_hist[-1]
            print(
                "Step %s, loss: %s, time elapsed: %s seconds"
                % (step, loss_value, time.time() - t0)
            )

    return params_t, torch.stack(loss_hist)


def gen_random_imgs(
    Ndata: int, height: int, width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Generates random images

    Args:
        Ndata (int): Number of images
        height (int): Image height
        width (int): Image width

    Returns:
        np.ndarray: Array of shape (Ndata, height, width) containing image values
        np.ndarray: Array of shape (Ndata) containing image amplitudes
    """
    random_images = np.zeros((Ndata, height, width))
    amplitude_vals = np.zeros(Ndata)

    for i in range(0, Ndata):
        random_images[i] = np.random.rand(height, width)
        amplitude_vals[i] = np.sum(random_images[i] ** 2) ** 0.5
    return random_images, amplitude_vals


def train(states_diff):
    """Trains model with forwards diffusion states

    Args:
        states_diff (np.ndarray): Diffusion states through T+1 of dim (T+1, Ndata, n**2)
    """
    print("Training")
    inputs_T = diffModel.HaarSampleGeneration(Ndata, seed=22)
    model = QDDPM(n=n, na=na, T=T, L=L)
    model.set_diffusionSet(states_diff)

    params_total = []
    loss_hist_total = []

    for t in range(T - 1, -1, -1):
        params_tot = np.zeros((T, 2 * (n + na) * L))
        print("  Training step: %d" % t)
        for tt in range(t + 1, T):
            params_tot[tt] = np.load("%s/params_t%d.npy" % (dir, tt))

        params, loss_hist = Training_t(model, t, inputs_T, params_tot, Ndata, epochs)

        np.save("%s/params_t%d" % (dir, t), params.detach().numpy())
        np.save("%s/loss_t%d" % (dir, t), loss_hist.detach().numpy())

    params_tot = np.zeros((T, 2 * (n + na) * L))
    loss_tot = np.zeros((T, epochs))

    for t in range(T):
        params_tot[t] = np.load("%s/params_t%d.npy" % (dir, t))
        loss_tot[t] = np.load("%s/loss_t%d.npy" % (dir, t))

    np.save("%s/params_total_%dNdata_%dEpochs" % (dir, Ndata, epochs), params_tot)
    np.save("%s/loss_tot_%dNdata_%dEpochs" % (dir, Ndata, epochs), loss_tot)

def test():
    print("Testing")
    diff_hs = np.linspace(0.5, 4.0, T)
    diffuse_square = np.zeros((Ndata, 2**n)) + 1j * np.zeros((Ndata, 2**n))
    for i in range(0, Ndata):
        diffuse_square[i] = np.array(source_values / (12**0.5)) + 1j * np.zeros(2**n)

    X = torch.from_numpy(diffuse_square)

    Xout = np.zeros((T + 1, Ndata, 2**n), dtype=np.complex64)
    Xout[0] = X

    for t in range(1, T + 1):
        Xout[t] = diffModel.set_diffusionData_t(t, X, diff_hs[:t], seed=t).numpy()


    np.save("%s/Xout_test_image" % (dir), Xout)

    # use model to predict what original image was from diffused image
    test_data_T20 = torch.tensor(Xout[T], dtype=torch.complex64)

    # Created during training
    params_tot = np.load("%s/params_total_%dNdata_%dEpochs.npy" % (dir, Ndata, epochs))

    inputs_te = diffModel.HaarSampleGeneration(Ndata, seed=22)

    model = QDDPM(n=n, na=na, T=T, L=L)
    # Turn image data into qubits
    random_images, amplitude_vals = gen_random_imgs(Ndata, n, n)
    random_images_qubits = np.zeros((Ndata, 2**n)) + 1j * np.zeros((Ndata, 2**n))
    for i in range(0, Ndata):
        random_images_qubits[i] = np.ravel(random_images[i] / amplitude_vals[i] + 0j)
    data_te = model.backDataGeneration(
        torch.from_numpy(random_images_qubits), params_tot, Ndata
    )[:, :, : 2**n].numpy()

    np.save("%s/test_backwards_gen" % (dir), data_te)

    # Get the nxn array of the image from data_te
    final_output_nxn = np.zeros((T + 1, Ndata, n, n))

    for z in range(0, T + 1):
        final_output_flattened = data_te[z]
        final_output_flattened = np.abs(final_output_flattened)
        multiplier = np.max(final_output_flattened)
        for i in range(0, np.size(amplitude_vals)):
            final_output_flattened[:][i] *= amplitude_vals[i]
        # final_output_flattened = final_output_flattened.flatten()

        for i in range(0, n):
            for j in range(0, n):
                for nth_data in range(0, Ndata):
                    final_output_nxn[z][nth_data][i][j] = final_output_flattened[
                        nth_data
                    ][(i * n) + j]

    # Calculate mean square error

    mse_calc = np.zeros((T + 1, n**2))
    avg_backwards_vector = np.sum(data_te, axis=1) / Ndata

    orig_vals = np.sum(training_data, axis=0) / Ndata
    print("  MSE:")
    for i in range(0, T + 1):
        temp1 = 0
        for j in range(0, 2**n):
            temp1 += (np.abs(orig_vals[j] - avg_backwards_vector[i][j])) ** 2
        print("  T%d: %f"%(i,temp1))
        mse_calc[i] = temp1 / (n**2)


if __name__ == "__main__":
    args = parser.parse_args()
    if not os.path.isdir(args.dir) or not os.path.exists(args.dir):
        print("Invalid dir. Exiting")
        exit(1)
    dir = args.dir
    if args.g:
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        else:
            print("No GPU available. Torch will use CPU.")

    source_values = generate_source()

    training_data = generate_training(source_values, Ndata, 0.1)

    states_diff, diffModel = generate_diffusion_data(training_data)

    # train on diffusion data
    train(states_diff)
    # Diffuse our test image
    test()