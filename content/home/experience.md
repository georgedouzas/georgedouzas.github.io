+++
# Experience widget.
widget = "experience"  # See https://sourcethemes.com/academic/docs/page-builder/
headless = true  # This file represents a page section.
active = true  # Activate this widget? true/false
weight = 40  # Order that this section will appear.

title = "Professional   Experience"
subtitle = ""

# Date format for experience
#   Refer to https://sourcethemes.com/academic/docs/customization/#date-format
date_format = "Jan 2006"

# Experiences.
#   Add/remove as many `[[experience]]` blocks below as you like.
#   Required fields are `title`, `company`, and `date_start`.
#   Leave `date_end` empty if it's your current employer.
#   Begin/end multi-line descriptions with 3 quotes `"""`.
[[experience]]
  title = "Machine Learning Researcher"
  company = "University of Lisbon, Nova IMS"
  company_url = "https://www.novaims.unl.pt/magic/"
  location = "Portugal"
  date_start = "2018-09-01"
  date_end = ""
  description = """
  Designed, implemented and tested various new approaches to deal with the class imbalance problem. Research work focused on clustering based over-sampling methods that deal with the within-the-class imbalance problem. Additionally, Geometric SMOTE, an extension of the SMOTE algorithm, was proposed and implemented. The final publication presented results that show a significant improvement over SMOTE and its variations. Deep learning models, particularly Conditional Generative Adversarial Networks (CGANs), were also used as over-sampling methods with great success. The frameworks of the implementation were TensorFlow, Keras and PyTorch. Work is published in high impact machine learning journals. Implementation of the above algorithms was developed and made available as open source software. Work in progress includes comparative experiments between variations of CGANs as over-samplers as well as the investigation of novel algorithms in the context of reinforcement learning.
  """

[[experience]]
  title = "Machine Learning Engineer"
  company = "Tripsta"
  company_url = "https://www.tripsta.com/"
  location = "Greece"
  date_start = "2017-10-01"
  date_end = "2018-08-01"
  description = """
  Designed and implemented the main parts of the company's automated pricing system. These parts included machine learning estimators for the add-ons and the competitor’s prices as well as the application of metaheuristic algorithms for the budget multi-objective optimization problem. The training data of the various estimators were at the order of TB while the prediction time of the automated pricing system was required to be less than 100 msec for the incoming 50K requests/sec. The languages of the implementation were Python, Java and Scala while Spark, Dask, Scikit-Learn and jMetal were used as distributed data processing, machine learning and optimization frameworks/libraries.
  """

[[experience]]
  title = "Data Scientist"
  company = "Quantum Retail"
  company_url = "http://quantumretail.com/"
  location = "Remote"
  date_start = "2016-12-01"
  date_end = "2017-09-01"
  description = """
  Worked on demand forecasting and clustering for retail companies. Proposed and applied machine learning methods to improve the company’s main forecasting solution that was based on exponential smoothing of the time series data as well as adjustments guided by a seasonality curve. Boosting trees were selected as the final machine learning model. Applying feature extraction that integrated the business logic as well as applying extensive model hyperparameter tuning, the forecasting precision was improved by 30% compared to the original model.
  """

[[experience]]
  title = "Machine Learning Engineer"
  company = "CERN"
  company_url = "https://home.cern/"
  location = "Remote"
  date_start = "2016-05-01"
  date_end = "2016-09-01"
  description = """
  Developed the parallelization of various features for TMVA, the Toolkit for Multivariate Data Analysis with ROOT, as a part of a project funded by Google. ROOT is the main framework developed by CERN to deal with the big data processing, statistical analysis, visualization and storage of the massive amounts of data produced from the particle physics experiments. The legacy version was implemented in C++. The parallelized features included the application of brute-force and metaheuristic algorithms to the hyperparameter grid search of machine learning algorithms. The implementation was based on Python and Spark.
  """

[[experience]]
  title = "Scientific Software Engineer"
  company = "IRI"
  company_url = "https://www.iriworldwide.com/en-US"
  location = "Greece"
  date_start = "2014-10-01"
  date_end = "2016-05-01"
  description = """
  Member of the IRI's "Solutions and Innovation Team" (R&D) working on the company's transition towards Open Source and Elastic Computing. Participated in an agile team working on the migration of IRI’s main US "Price & Promo Analytics" Solution, generating more than $25M Annual Revenues, to Hadoop distributed storage and Spark cluster computing. Python was the core language of the implementation, but integration with R and Julia was performed to leverage special functionality. The legacy version was implemented in SAS. The main objectives of the project were the design of the parallelization schema, the enhancement of data manipulation with the use of distributed processing and the migration of the statistical modeling algorithms (regression mixed models). The final system was able to process 5 years of data for more than 300 categories containing 1 million products.
  """

[[experience]]
  title = "Co-owner & CTO"
  company = "Sports Performance Training"
  company_url = ""
  location = "Greece"
  date_start = "2009-09-01"
  date_end = "2013-09-01"
  description = """
  Co-owner and CTO of Sports Performance Training (SPT) startup. SPT provided consulting services to individual professional athletes and athletic organizations in multiple sports. Using standard ergometric measurements and a custom process through biometric sensors, a variety of signals were recorded and preprocessed. A combination of descriptive statistics, predictive modeling and domain knowledge was applied to adjust the training load of the athlete or team in the pre-competition period and to maximize performance in the competition period. The deliverable to the client included also training guidelines on individual basis that aimed to avoid sports specific injuries.
  """


+++
