# An open access tool for exploring machine learning model choice for battery life cycle prediction

This Showcase enables users to apply the simple linear regression models from Severson et al. Nature Energy (2019) to curated datasets on the Voltaiq Community server, as well as on custom data accessed via a search phrase. Train and test the Severson models, and predict cycle life for ongoing or new tests.

The notebook contained in this Showcase was further used to generate the insights from the following open-access publication:
[Schauser Nicole S., Lininger Christianna N., Leland Eli S., Sholklapper Tal Z. An open access tool for exploring machine learning model choice for battery life cycle prediction. Frontiers in Energy Research, 10 (2022) DOI: 10.3389/fenrg.2022.1058999](https://www.frontiersin.org/articles/10.3389/fenrg.2022.1058999)

Early and accurate battery lifetime predictions could accelerate battery R&D and product development timelines by providing insights into performance after only a few days or weeks of testing rather than waiting months to reach degradation thresholds. However, most machine learning (ML) models are developed using a single dataset, leaving unanswered questions about the broader applicability and potential impact of such models for other battery chemistries or cycling conditions. In this work, we take advantage of the open-access cycling performance data within the recently released Voltaiq Community to determine the extensibility of a highly cited feature-based linear ML model used for battery lifetime prediction. We find that the model is unable to extrapolate to different datasets, with severe model overfitting resulting in unphysical lifetime predictions of much of the unseen data. We further identify that the features engineered for this model are likely specific to the degradation mode for the original lithium iron phosphate (LFP) fast-charge dataset and are unable to capture the lifetime behavior of other cathode chemistries and cycling protocols. We provide an open access widget-based Jupyter Notebook script that can be used to explore model training and lifetime prediction on data within the Voltaiq Community platform. This work motivates the importance of using larger and more diverse datasets to identify ML model boundaries and limitations, and suggests training on larger and diverse datasets is required to develop data features that can predict a broader set of failure modes.

---

This repository was created by Voltaiq Community Edition. Please do not delete it or change its
sharing settings.
