# TrFedDis: Trusted Federated Disentangling Network for Non-IID Domain Feature

# Abstract:
Federated Learning (FL), as an efficient decentralized distributed learning approach, enables multiple institutions to collaboratively train a model without sharing their local data. Despite its advantages, the performance of FL models is substantially impacted by the domain feature shift arising from different acquisition devices/clients. Moreover, existing FL methods often prioritize accuracy without considering reliability factors such as confidence or uncertainty, leading to unreliable predictions in safety-critical applications. Thus, our goal is to enhance FL performance by addressing non-domain feature issues and ensuring model reliability. In this study, we introduce a novel approach named RFedDis (Reliable Federated Disentangling Network). RFedDis leverages feature disentangling to capture a global domain-invariant cross-client representation while preserving local client-specific feature learning. Additionally, we incorporate an uncertainty-aware decision fusion mechanism to effectively integrate the decoupled features. This ensures dynamic integration at the evidence level, producing reliable predictions accompanied by estimated uncertainties. Therefore, RFedDis is the FL approach to combine evidential uncertainty with feature disentangling, enhancing both performance and reliability in handling non-IID domain features. Extensive experimental results demonstrate that RFedDis outperforms other state-of-the-art FL approaches, providing outstanding performance coupled with a high degree of reliability.
