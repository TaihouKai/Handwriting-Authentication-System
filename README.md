# Handwriting-Authentication-System

Wang Pengfei, Liu Fangrui, Zheng Qiushi, Google LLC

#   License

> We currently DO NOT offer any license.

#   Dev Log

* Tasks
    * [x] Image Processing Stage
        * [x] Detection
            * [x]   EdgeBox -proposal extraction
            * [x]   ContourBox -proposal extraction
            * [x]   AlexNet@MNIST classification
        * [x] Feature Extraction
            * [x]   SIFT / ORB
        * [x] Module test
            * [x]   Detector + Classifier
            * [x]   Detector + Classifier + Extractor
    * [ ] Backend
        * [ ] Platform (container / API)
        * [x] Client UI / Algorithm re-implementation
        * [x] Data Collection
     * [ ] Data Analysis
        * [ ] What percentage of features will be authenticated by users during logging in, on average?
        * [ ] About the question above, does scenario matter? How?
        * [ ] ... other discoveries?
        * [ ] Therefore, what threshold(s) should be set to distinguish "authorized" and "unauthorized" logging-in attempts?
        * [ ] (Optional) Success rate for OCR?

#   MEMO

> Forget this for now. They are out-dated.

* Add this function: relative position of the characters need to be checked as well!
   * Comment this in the code. We won't demonstrate this function in the demo.

* Add this function: return registered position to the user. It should be an optional function. 
