# Floor-plan-analysis-using-digital-image-processing-techniques
This project is about using computer vision - specifically Digital Image Processing Techniques - to analyze architectural floor plans and thereby, extract the area and the number of rooms. This would allow for a significant reduction in the time taken to manually inspect floor plans to then calculate property valuations. 

For a quick overview of the project and details about the implementation, please see the PDF I have created:

https://drive.google.com/file/d/1_lXnquwGzTpXTxfu77swtlSrJb1qrOoi/view?usp=sharing

# 1. Project Overview
Architectural floor plan analysis has been under research for a long time under the
branch of computer vision called automated document analysis. A floor plan represents
the relationship between rooms and spaces and their structures within a property. They
play a crucial role in allowing people to quickly understand the indoor space of a
property. Floor plans are drawn to scale using an architectural drawing tool called CAD
to produce an image in vectors graphics format. However, when these plans are printed
out, they are rasterized. This results in the loss of essential structural and semantic
information. Recovering this lost information can be very useful for various purposes.

Council tax is a local taxation system used to determine the tax on domestic properties.
Properties are analyzed in order to determine their council tax band and hence provide a
property valuation. The process of analysis is done by inspecting floor plans and
obtaining the number of different types of rooms, the area of the property, its age, group,
and type. If the property is flat or maisonette, then the area measured is called the
Effective Floor Area - EFA whereas if the property is a house or a bungalow, the area
measured is called the Reduced Covered Area - RCA. Broadly speaking, the EFA is the
area measured externally around the outermost walls. RCA is the area measured
internally around the outermost walls. Currently, the process of inspecting floor plans is
performed manually bu human subjects called caseworkers. The manual inspection of
floor plans is a heavy, time consuming, and resource-consuming task given the fact that
there is a large number of properties. Caseworkers have the capacity to inspect only
about 6 floor plans per day.

This project aims to address the above problem by providing a system capable of
automatically extracting vital information from the floor plan which is required o value
them.

Firstly, using digital image processing techniques including morphological
transformations, image thresholding, contour detection, corner detection, and
rectangular approximation, we present algorithms for calculating the Reduced Covered
Area, Effective Floor Area, and detecting room boundaries. This is achieved by
extracting the skeleton of the floor plan and then contours of the internal and external
boundaries of the outermost walls of the floor plan. These contours are used to calculate
the different kinds of areas. 

The above be found under the following notebooks:
1. final > Effective Floor Area.ipynb
2. final > Reduced_covered_Area.ipynb
3. final > Room Boundary detection.ipynb
4. final > Room detection.ipynb

Secondly, this project focuses on detecting the number of
different types of rooms on the floor plan. Floor plans contain text labels representing the
type and location of each room. By utilizing a neural-network-based robust
state-of-the-art text detection model called EAST and an LSTM based text recognition
library - Tesseract 4, the text on the floor plan is obtained which is then used to count the
different kinds of rooms.

The current system was tested on floor plans for domestic properties since this paper
aims to provide a tool for efficient council tax calculation. This project focuses on area
calculation and room count calculation since these are the parameters that can be
extracted by inspecting floor plans and require the most amount of time when inspected
manually.

The above be found under the following notebooks:
1. final > Text Detection and Recognition.ipynb

# 2. Analysis of strengths
In this project, an extensive amount of time was spent on research. The previous
techniques were explored and analyzed. This helped me understand the methods which
were used when document analysis was a relatively newer concept. Most previous
papers were based on low-level image processing techniques - such as edge detection,
corner detection, hough line transformation - in order to segment the image and also
perform structural analysis. Performing an extensive literature review also helped me
understand why newer techniques have moved towards deep learning-based
approaches.

In my project, I used low-level image processing techniques to perform structural
analysis. I performed morphological transformations and contour detection which were
used to calculate the reduced covered area and the effective floor area of the floor plans.
Since the thresholds were fixed for binarizing the image and kernel sizes were fixed
while performing morphological transformations, this method worked will for floor plans
with certain structural conventions while calculation the area. In particular, it provided
accurate results for floor plan images with thick outermost walls.

I also performed rigorous research on understanding Optical Character Recognition and
how its techniques have evolved over time. The most recent techniques are based on
neural networks. I also took up a course for convolutional neural networks (delivered by
Andrew Ng) in order to understand in-depth how deep learning is used in computer
vision. The results obtained by the robust text detection model called EAST - Efficient
and Accurate Scene - and text recognition library - Tesseract 4 - were highly accurate.
The recognized text was then passed through a spell check and then finally used to
count the number of different types of rooms on the floor plan. I was able to obtain an
accuracy of 71 % after testing this on the dataset I had created.

# 3. Analysis of weaknesses
Although the techniques explored provide good results, this project does have certain
weaknesses. For area calculation and then contour detection, morphological
transformations were performed using fixed kernel sizes and fixed thresholds regardless
of the input floor plan image. The floor plan images have highly varying structural
conventions in terms of wall thickness, symbol conventions, and even textual structure to
represent room types and their locations.

While calculating the effective floor area, a prerequisite step is to perform the
morphological opening of the image and extract the skeleton of the floor plan. This
skeleton comprises the bearing walls. The kernel size used for this operation is fixed
for all images. If the floor plan does not have walls thicker than a certain amount, then
the walls end up being eroded too during the morphological opening. Thus, the contours
detected for the EFA calculation are not accurate for all kinds of floor plans.
Since the kernel sizes and thresholds are fixed, it is preferable to use deep learning
techniques and create a model that is able to learn and identify bearing walls, doors, and
windows. I performed rigorous research to look for datasets that satisfy the needs of this
project. Due to the limitations of open-source floor plans specific to the requirements, I
used low-level image processing techniques.

Lastly, the EAST model-based text detection was highly accurate for most floor plan
images with varying structures and varying fonts for room labels. The detected text was
then recognized using the Tesseract OCR which was then passed through a spell check.
Often, the recognized text was misspelled and was unable to be correctly spelled even
after using a spell checker. This resulted in the text being discarded despite being
detected correctly. Sometimes there are different names for representing the same type
of room for example family room instead of bedroom or utility room instead of store
room. This also leads to some inaccuracy while counting different kinds of rooms.

# 4. Presentation of Possibilities for Further Work
Digital image processing techniques have proven to work well however, they do have
certain limitations. If presented with a hand-drawn floor plan or floor plans with thin walls,
the given methodology is not likely to not work very well. Deep learning-based
techniques can be used to provide improved and accurate results. Deep learning will
allow the learning of different elements on the floor plan and hence recognize different
elements despite their graphical conventions. In order to detect room boundaries,
bearing walls, doors, and windows, a multi-task neural network can be used similar to
the one presented in. Previously, this has been difficult to achieve due to the limited
dataset available for deep learning-based floor plan analysis. However, if a dataset is
created specific to the problem at hand, it can result in the development of a highly
effective model for floor plan analysis.

Moreover, more tools can be used for text detection and recognition and compared with
the current methodology to check for improved accuracy.

Lastly, deep learning for floor plans can be used to detect whether a floor plan exists in
an image as well. This is an important step before property valuations are performed.

# 5. Critical analysis of the relationship between theoretical and practical work
Theoretical knowledge is important to gain a strong idea of the concepts but once the
theory is implemented in practice, the intuition behind every step begins to make sense.
In theory, morphological transformations are able to perform the exact tasks required to
achieve the results needed for this project. However, after practical experiments, I
realized that deep learning techniques may work better. Considering all of this, in
practical implementations, obtaining ideal results is also quite difficult especially since
the input floor plan images have highly varying conventions.

# 6. Awareness of Legal, Social Ethical Issues and Sustainability
a. Legal Issues
Since this project aims to provide convenience for property valuations, it can
possibly have legal implications. This system needs to be highly accurate in order
to provide results so that properties are valued fairly and efficiently.
b. Social Issues
The legal issues for this project are directly connected to social issues. If the area
is not calculated correctly or if the room labels are not recognized correctly, the
council tax band calculation will be inaccurate. Hence the tax needed to pay for
the particular property will not be presented correctly. This can easily lead to
misunderstandings and social problems among tax collectors and payers.
c. Sustainability
The system presented in this paper and its potential future works is highly
sustainable for the future and further development of automation processes in
property valuations. This system can also be used for 3D modeling of floor plans
and creating vectorized big data of floor plans, and reconstruction or modification
of current floor plans.
