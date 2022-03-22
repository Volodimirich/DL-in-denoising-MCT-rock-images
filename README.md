## This is the repository of the final project for the Machine Learning 2022 course in Skoltech.

In this project results of the article [“Deep learning in denoising of micro-computed tomography images of rock samples”](https://doi.org/10.1016/j.cageo.2021.104716) are reproduced. Two approaches to remove the noise from CT rock images are utilised: supervised model RedNET, and self-supervised model DIP. 

DIP model and following instructions are located in [this folder](https://github.com/Volodimirich/DL-in-denoising-MCT-rock-images/tree/main/DIP).

Images denoised with DIP were used to train RedNET as the target. Different loss functions were used to evaluate the supervised model and obtain best denoising performance.

As improvements to the original article, new spectral residual-based loss SR-SIM and DnCNN achitecture are proposed. 

YAML configuration files are located in the [configs folder](https://github.com/Volodimirich/DL-in-denoising-MCT-rock-images/tree/main/configs)

## License
MIT

## Contact
Vladimir.Shaposhnikov@skoltech.ru

dmitry.artemasov@skoltech.ru


