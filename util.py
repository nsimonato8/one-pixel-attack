import numpy as np
import pandas as pd

def prepare_fair_samples(x: np.ndarray, y: np.ndarray, label_names: list, sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
        Returns a dataset for which each label has the same number samples.
    """
    image_fair_sample, label_fair_sample = [], []
    batch_size = int(sample_size / len(label_names))
    for i, _ in enumerate(label_names):
        mask = np.reshape(y, (y.shape[0],)) == i
        y_masked = y[mask]
        x_masked = x[mask]
        sample_mask = np.random.permutation(np.concatenate((np.array(batch_size*[True]), np.array((len(y_masked) - batch_size)*[False]))))
        image_fair_sample.append(x_masked[sample_mask])
        label_fair_sample.append(y_masked[sample_mask])
    return np.concatenate(image_fair_sample), np.concatenate(label_fair_sample)

def filter_valid_samples(model, x, y_true) -> tuple[np.ndarray, np.ndarray]:
    """
        Filters only the correct samples from the dataset, that is, the samples for which the model 
        returns the correct label.
    """
    y_pred = np.argmax(model.predict(x), axis=1)
    y_pred = np.reshape(a=y_pred, newshape=(len(x),))
    y_true = np.reshape(a=y_true, newshape=(len(y_true),))
    mask = y_pred == y_true
    return x[mask], y_true[mask]

def generate_conf_matrix(classifier, original_labels: np.ndarray, adversarial_labels: np.ndarray, class_names: list[str], name: str, labels: list = list(range(10)), today: str=None) -> None:
    """
        Returns the confusion matrix for the results of the classifier.
    """    
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot as plt
    import seaborn as sn
    conf_matr = confusion_matrix(y_true=original_labels, y_pred=adversarial_labels, labels=labels)
    plt.figure(figsize=(20,20))
    ax = sn.heatmap(conf_matr, annot=True, cbar=True, square=True, 
                    xticklabels=class_names, yticklabels=class_names, 
                    linewidth=.25, annot_kws={"size": 18})
    title= name[0].upper() + name[1:]+" heatmap"
    ax.set(title=title, xlabel="True labels", ylabel="Predicted labels")
    if today is not None:
        plt.savefig(f"results/{today}/{name}_{classifier.name}_confusion_matrix.png")
    plt.show()
    pass

def show_example_images(original_image, original_label: str, adversarial_image, adversarial_label: str, class_names: list[str]) -> None:
    """
        Prints in sequence: an image, its adversarial counterpart and the perturbation.
        Also prints the labels.
    """
    from matplotlib import pyplot as plt
    import seaborn as sn
    fig, axes = plt.subplots(1, 3, figsize=(20, 20))
    ax = axes.flat
    sn.set(font_scale=1)
    ax[0].imshow(original_image)
    ax[0].set_title("Original")
    ax[0].set_xlabel(f"Label: {class_names[original_label]}")
    ax[1].imshow(adversarial_image)
    ax[1].set_title("Adversarial")
    ax[1].set_xlabel(f"Label: {class_names[adversarial_label]}")
    ax[2].imshow(adversarial_image - original_image)
    ax[2].set_title("Perturbation")
    plt.show()
    pass

def plot_ROC_curve(classifiers: list, name:str, images: list, labels: list, class_names: list[str], today: str=None) -> None:
    """
        Prints the One-vs-All ROC curve.
    """
    # Subdivide the plot space in a row per model
    # Get the ROC curve for each model and plot it
    # Remeber to plot the labels
    from matplotlib import pyplot as plt
    from sklearn.metrics import RocCurveDisplay
    from sklearn.preprocessing import LabelBinarizer
    from itertools import cycle
    
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = cycle(["aqua", "darkorange", "cornflowerblue", "red", "white", "orange"])
    label_binarizer = LabelBinarizer().fit(labels)
    y_onehot_test = label_binarizer.transform(labels)
    for mod in classifiers:
        y_score = mod.predict(images)
        for class_id, color in zip(range(len(class_names)), colors):
            RocCurveDisplay.from_predictions(
                y_onehot_test[:, class_id],
                y_score[:, class_id],
                name=f"ROC curve for {class_names[class_id]}",
                color=color,
                ax=ax
            )
    if today is not None:
        plt.savefig(f"results/{today}/{name}_{classifiers[0].name}_roc_curve.png")
    plt.show()
    pass

def print_distribution_boxplots(distr: dict, today: str=None) -> None:
    """
        Prints the boxplot of the distributions contained in distr.
        Keyword arguments:
            distr -- a dictionary made of tuples (str, ndarray) where the first element
                     is the name of the distribution, and the second element is the 1D array 
                     containing the samples from the distribution.
            today -- a string with today's date in the YYYY-MM-DD format. If passed, the boxplot
                     will be saved in a .png file in the folder 'results/{today}', otherwise defaults
                     to None.
    """
    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()
    ax.boxplot(distr.values())
    ax.set_xticklabels(distr.keys())
    if today is not None:
        plt.savefig(f"results/{today}/reconstructed_loss_distribution.png")
    plt.show()
    pass