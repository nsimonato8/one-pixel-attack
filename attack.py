import numpy as np
import pandas as pd
import matplotlib as plt
from functools import partial
from scipy.optimize import differential_evolution

def predict_classes(xs: np.ndarray, img: np.ndarray, target_class: int, model, minimize=True) -> np.ndarray:
    imgs_perturbed = np.reshape(a=perturb_image(xs, img), newshape=(1,) + img.shape)
    predictions = model(imgs_perturbed)[:, int(target_class)]
    # This function should always be minimized, so return its complement if needed
    return predictions if minimize else 1 - predictions

def attack_success(x, img, target_class: int, model, convergence: float, targeted_attack=False, verbose=True) -> bool:
    if convergence > 1.:
        return True
    attack_image = np.reshape(a=perturb_image(x, img), newshape=(1,) + img.shape)
    confidence = model(attack_image)[0]
    predicted_class = int(np.argmax(confidence))

    # If the prediction is what we want (misclassification or
    # targeted classification), return True
    if verbose:
        print('Confidence:', confidence[int(target_class)])
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
        return True
    return False

def attack_success_(targeted, target_class, model, img):
    return partial(attack_success, targeted_attack=targeted, img=img, target_class=target_class, model=model)

def perturb_image(xs: np.ndarray, img: np.ndarray) -> np.ndarray:
    adv_img = img.copy()
    x_pos, y_pos, *rgb = xs
#   rgb = [x / 255. for x in rgb]
    adv_img[x_pos.astype(int), y_pos.astype(int), :] = rgb
    return adv_img


def plot_image(image: np.ndarray, label_true=None, class_names: list[str]=None, label_pred=None) -> None:
    if image.ndim == 4 and image.shape[0] == 1:
        image = image[0]

    plt.grid()
    plt.imshow(image.astype(np.uint8))

    # Show true and predicted classes
    if label_true is not None and class_names is not None:
        labels_true_name = class_names[label_true]
        if label_pred is None:
            xlabel = "True: " + labels_true_name
        else:
            # Name of the predicted class
            labels_pred_name = class_names[label_pred]

            xlabel = "True: " + labels_true_name + "\nPredicted: " + labels_pred_name

        # Show the class on the x-axis
        plt.xlabel(xlabel)

    plt.xticks([])  # Remove ticks from the plot
    plt.yticks([])
    plt.show()  # Show the plot
    pass


def plot_images(images, labels_true, class_names: list[str], labels_pred=None,
                confidence=None, titles=None) -> None:
    assert len(images) == len(labels_true)

    # Create a figure with sub-plots
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))

    # Adjust the vertical spacing
    hspace = 0.2
    if labels_pred is not None:
        hspace += 0.2
    if titles is not None:
        hspace += 0.2

    fig.subplots_adjust(hspace=hspace, wspace=0.0)

    for i, ax in enumerate(axes.flat):
        # Fix crash when less than 9 images
        if i < len(images):
            # Plot the image
            ax.imshow(images[i])

            # Name of the true class
            labels_true_name = class_names[labels_true[i]]

            # Show true and predicted classes
            if labels_pred is None:
                xlabel = "True: " + labels_true_name
            else:
                # Name of the predicted class
                labels_pred_name = class_names[labels_pred[i]]

                xlabel = "True: " + labels_true_name + "\nPred: " + labels_pred_name
                if (confidence is not None):
                    xlabel += " (" + "{0:.1f}".format(confidence[i] * 100) + "%)"

            # Show the class on the x-axis
            ax.set_xlabel(xlabel)

            if titles is not None:
                ax.set_title(titles[i])

        # Remove ticks from the plot
        ax.set_xticks([])
        ax.set_yticks([])

    # Show the plot
    plt.show()
    pass


def attack_stats(df: list, models: list) -> pd.DataFrame:
    df = pd.DataFrame(df, columns=["model_name", "pixel_count", "img", "actual_class", "predicted_class", "success", "cdiff", "prior_probs", "predicted_probs", "perturbation"])
    stats = []
    for model in models:
        val_accuracy = len(df[df['actual_class'] == df['predicted_class']]) / len(df)
        success_rate = len(df[df["success"] == 1]) / len(df)        
        stats.append([model.name, val_accuracy, 1, success_rate])

    return pd.DataFrame(stats, columns=['model', 'accuracy', 'pixels', 'attack_success_rate'])


def attack(img: np.ndarray, model, mp: int, true_label: int, target: int=None, pixel_count=1,
           maxiter: int=75, popsize: int=400, verbose: bool=False, plot: bool=False) -> list:
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else true_label
    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    if len(img.shape) == 3:
        bounds = [(0, img.shape[0]), (0, img.shape[1]), (0, 256), (0, 256), (0, 256)]
    else:
        bounds = [(0, img.shape[0]), (0, img.shape[1]), (0, 256)]
        
    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))
    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        func=predict_classes,        
        bounds=bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1,
        args=(img, target_class, model),
        callback=attack_success_(img=img, targeted=targeted_attack, target_class=target_class, model=model),
        polish=False, disp=verbose, workers=mp) 
    
    # Calculate some useful statistics to return from this function
    attack_image = perturb_image(attack_result.x, img)
    prior_probs = model.predict(np.array([img]))[0]
    predicted_probs = model.predict(np.array([attack_image]))[0]
    predicted_class = np.argmax(predicted_probs)
    actual_class = true_label
    success = predicted_class == target_class if targeted_attack else predicted_class != actual_class
    cdiff = prior_probs[actual_class] - predicted_probs[actual_class]

    # Show the best attempt at a solution (successful or not)
    if plot:
        plot_image(attack_image, actual_class, class_names, predicted_class)

    return [model.name, pixel_count, img, actual_class, predicted_class, success, cdiff, prior_probs,
            predicted_probs, attack_result.x]

def attack_all(models, test: tuple[np.ndarray, np.ndarray], mp: int, target: int, class_names: list[str], maxiter:int =75, pixels=(1,3,5), popsize: int=400, verbose: bool=False) -> list:
    results = []
    targeted = target is not None
    img_samples, label_samples = test # x_test, y_test = test
    
    for model in models:
        model_results = []
        
        for pixel_count in pixels:
            for i, img in enumerate(img_samples):
                if verbose:
                    print(model.name, '- image', img, '-', i + 1, '/', len(img_samples))
                
                if targeted:
                    target_ = target
                    if verbose:
                        print('\tcAttacking with target', class_names[target])
                    if target == label_samples[i]:
                        continue
                else:
                    target_= None
                    
                result = attack(img=img_samples[i], 
                                model=model, 
                                target=target_,
                                pixel_count=pixel_count,
                                true_label=label_samples[i],
                                maxiter=maxiter, 
                                popsize=popsize,
                                mp=mp,
                                verbose=verbose)
                
                model_results.append(result)

        results += model_results
    return results
