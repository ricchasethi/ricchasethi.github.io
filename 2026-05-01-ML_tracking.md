---
layout: post
title: "OncoAgent: Medical AI decision system for different subtypes of Glioblastoma Multiform (GBM)"
date: 2026-05-01
---
# Prediction to Decision: Explainable AI system for Glioblastoma (Week 1)
Over past week I started working on a project that would build a clinical decision system for prediction of subtype of Glioblastoma using bulk RNA sequencing and also explain what that prediction would mean for the patient. The architecture would look like:

Patient Data (bulk RNA sequencing) --> PyTorch Model --> Subtype Prediction --> Retreival augmented generation (literature) --> Agent --> Explainable Decision Output 

OncoAgent is a production-style AI decision system built to tackle this classification problem end-to-end. It will focus on data pipelines, reproducibility, deployment, usage of Retreivel Augemented Generation and agent workflow for explainability.

The goal here is to take research style model and productionise it into a usable system. Rather than focusing on development of ML model, here, I will focus on explaining the key steps in converting this prototype model to a productionised decision system. This productionised system can help clinicians in deciding the course of treatment for Glioblastoma patients.

This week I will discuss workflow for data ingestion, training a simple neural network using PyTorch, experiment tracking using MLFlow and exposure of model via FastAPI service. At the end of this post we will have an API that can be plugged into hospital dashboards, electronic health systems and clinical decision support tools. I really liked an analogy used by ChatGPT to explain what is an API. Think it as a restaurant waiter that does not cook himself but takes your (customer) order by bringing menu, suggest food items as per your taste, communicate to chef in the kitchen to prepare that order, and serves that order to your table. So, in this project, FastAPI will create an API (waiter) that will take in patient's gene expression (food order), communicate to trained model for predicting the subtype of Glioblastoma (chef) and deliver a score that gives a probability of patient classified to a particular subtype (your food).

## GBM subtype classifier on TCGA bulk RNA sequencing data

Glioblastoma (GBM) is one of the most aggressive primary brain tumours with a median survival of just 15 months. One of the well-established findings in GBM biology is that the disease is not homogeneous — it can be divided into at least four transcriptional subtypes: **Classical**, **Mesenchymal**, **Proneural**, and **Neural**. These subtypes carry different prognoses and may respond differently to treatment. Automated, accurate subtype classification from bulk RNA sequencing data could be a meaningful step toward personalised neuro-oncology.

