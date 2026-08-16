# ML Theory Notes: Instance-Based Learning & Clustering

สรุปทฤษฎี บทเรียน และตัวอย่างการคำนวณแบบเจาะลึกของ 3 อัลกอริทึมหลักใน Machine Learning: **KNN (K-Nearest Neighbors)**, **KNN Regression** และ **K-Means Clustering**

---

## 📌 ภาพรวม (Overview)

ทั้งสามอัลกอริทึมนี้มีแก่นแท้ในการทำงานเดียวกันคือ **"การวัดระยะห่างระหว่างจุดข้อมูลบนพิกัดเวกเตอร์ (Distance-Based Metrics)"** แต่มีเป้าหมายและการประยุกต์ใช้งานที่แตกต่างกัน:

| อัลกอริทึม | ประเภทงาน (Task) | การเรียนรู้ (Learning Type) | ผลลัพธ์ (Output) |
| :--- | :--- | :--- | :--- |
| **KNN** | Classification | Supervised / Instance-Based | หมวดหมู่/คลาส (Class Label) จาก Majority Vote |
| **KNN Regression** | Regression / Prediction | Supervised / Instance-Based | ค่าตัวเลข (Continuous Value) จาก Plain/Weighted Average |
| **K-Means** | Clustering | Unsupervised / Partitioning | กลุ่มข้อมูล (Clusters) จาก Centroid |

---

## 📖 เนื้อหาและทฤษฎี (Theoretical Breakdown)

### 1. K-Nearest Neighbor (KNN) — Classification

- **ประเภทการเรียนรู้:** Instance-Based / Lazy Learning (ไม่สร้าง Model Parameter ล่วงหน้า แต่ใช้ข้อมูลฝึกทั้งหมดทำนาย ณ เวลาจริง)
- **หลักการ:** "สิ่งที่คล้ายกันมักจะอยู่ใกล้กัน" (Objects near each other share similar properties)
- **กระบวนการ:**
  1. รับจุดข้อมูลใหม่ ($x_{test}$)
  2. คำนวณระยะห่างกับจุดข้อมูลฝึกทั้งหมด ($x_i$)
  3. เรียงลำดับระยะห่างจากน้อยไปมาก
  4. เลือก $k$ เพื่อนบ้านที่ใกล้ที่สุด
  5. โหวตเสียงข้างมาก (**Majority Voting**) เพื่อทำนายคลาส
- **สูตรระยะทาง (Euclidean Distance):**
  $$d(A, B) = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}$$
- **ผลกระทบของค่า $k$:**
  - $k$ น้อย: ความยืดหยุ่นสูง (Flexible), ไวต่อ Noise, เสี่ยง Overfitting
  - $k$ มาก: โมเดลเรียบเสถียร (Stable), เสี่ยง Underfitting
- **ข้อจำกัด:** ใช้หน่วยความจำสูง, ทำนายช้าเมื่อชุดข้อมูลใหญ่, ประสิทธิภาพลดลงในมิติสูง (*Curse of Dimensionality*)

---

### 2. KNN Regression

- **ประเภทการเรียนรู้:** Supervised Learning สำหรับการพยากรณ์ค่าตัวเลขต่อเนื่อง
- **หลักการ:** อาศัยค่าเป้าหมาย (Target Values) ของเพื่อนบ้าน $k$ จุดที่ใกล้ที่สุด
- **รูปแบบการคำนวณ:**
  1. **Plain Average (ค่าเฉลี่ยธรรมดา):**
     $$\hat{y} = \frac{1}{k} \sum_{i=1}^{k} y_i$$
  2. **Weighted Average (ค่าเฉลี่ยถ่วงน้ำหนักตามระยะทาง):**
     $$\hat{y} = \frac{\sum_{i=1}^{k} w_i y_i}{\sum_{i=1}^{k} w_i} \quad \text{โดยที่} \quad w_i = \frac{1}{d(x, x_i)^2 + \epsilon}$$
- **ตัวอย่างการประยุกต์ใช้งาน:** การประเมินราคาอสังหาริมทรัพย์, การพยากรณ์ราคาหุ้น, อุณหภูมิ, ปริมาณการใช้พลังงาน

---

### 3. K-Means Clustering

- **ประเภทการเรียนรู้:** Unsupervised Learning (ไม่มี Label สำหรับฝึก)
- **หลักการ:** การจัดกลุ่มข้อมูลออกเป็น $K$ กลุ่ม โดยมุ่งเน้นการลดระยะห่างระหว่างจุดข้อมูลกับจุดศูนย์กลางกลุ่ม (**Centroid**)
- **ขั้นตอนการทำงาน (Algorithm Steps):**
  1. **Initialization:** กำหนดจำนวนกลุ่ม $K$ และสุ่ม/เลือกจุด Centroid เริ่มต้น $K$ จุด
  2. **Cluster Assignment:** คำนวณระยะห่างระหว่างจุดข้อมูลทุกจุดกับ Centroid แล้วจัดจุดข้อมูลเข้ากลุ่ม Centroid ที่ใกล้ที่สุด
  3. **Centroid Update:** คำนวณจุดศูนย์กลางใหม่ของแต่ละกลุ่ม โดยใช้ค่าเฉลี่ย (Mean) ของสมาชิกในกลุ่มนั้น
  4. **Convergence Check:** ทำซ้ำขั้นตอน 2–3 จนกระทั่ง Centroid ไม่มีการเปลี่ยนแปลงตำแหน่งอีกต่อไป (Converged)

---

## 🧮 ตัวอย่างการคำนวณ (Numerical Examples)

### ตัวอย่างที่ 1: KNN Classification ($k=3$)
**ข้อมูลฝึก (Training Data):** $A(2,3), B(3,5), C(6,8), D(7,6), E(5,4)$  
**Query Point ($P_1 = (4, 5)$):**

| จุด | พิกัด | Class | ระยะทาง Euclidean ($d$) | ผลการเลือก ($k=3$) |
| :---: | :---: | :---: | :---: | :---: |
| **B** | (3, 5) | B | $1.000$ | Select (Rank 1) |
| **E** | (5, 4) | A | $1.414$ | Select (Rank 2) |
| **A** | (2, 3) | A | $2.828$ | Select (Rank 3) |
| **D** | (7, 6) | B | $3.162$ | - |
| **C** | (6, 8) | B | $3.606$ | - |

- **สรุปผลโหวต:** Class A = 2 เสียง, Class B = 1 เสียง
- **คำตอบ:** ทำนายเป็น **Class A**

---

### ตัวอย่างที่ 2: Weighted KNN Regression (พยากรณ์ราคาบ้าน, $k=5$)
| Index | ระยะทาง ($d$) | ราคา ($y$ - พันบาท) | น้ำหนัก $w_i = \frac{1}{d^2 + 10^{-5}}$ |
| :---: | :---: | :---: | :---: |
| **7** | 2.00 | 1,200 | 0.2500 |
| **10** | 3.00 | 1,250 | 0.1111 |
| **9** | 7.00 | 1,150 | 0.0204 |
| **3** | 8.00 | 1,300 | 0.0156 |
| **6** | 13.00 | 1,350 | 0.0059 |
| **รวม** | - | - | **$\sum w_i = 0.4029$** |

$$\hat{y} = \frac{(0.2500 \times 1200) + (0.1111 \times 1250) + (0.0204 \times 1150) + (0.0156 \times 1300) + (0.0059 \times 1350)}{0.4029} \approx \mathbf{1,217.59 \text{ พันบาท}}$$

---

### ตัวอย่างที่ 3: K-Means Clustering ($K=2$)
**ข้อมูล:** $A(2,3), B(3,4), C(4,5), D(8,8), E(9,9), F(8,10)$  
**กำหนด Centroid เริ่มต้น:** $C_1 = A(2,3)$, $C_2 = E(9,9)$

**รอบที่ 1 (Iteration 1):**
- **Cluster 1 Assignment:** $A, B, C$
- **Cluster 2 Assignment:** $D, E, F$
- **ปรับ Centroid ใหม่:**
  - $C_1^{new} = \text{mean}(A, B, C) = \mathbf{(3.00, 4.00)}$
  - $C_2^{new} = \text{mean}(D, E, F) = \mathbf{(8.33, 9.00)}$
- **สถานะ:** ลู่เข้าสมบูรณ์ (Converged) ภายใน 1 รอบ

---

## 👨‍🏫 อ้างอิง & ผู้จัดทำ (Credits)

- **เนื้อหา:** Machine Learning Lecture Notes[cite: 1]
- **ผู้สอน/ผู้จัดทำ:** Dr. Olarik Surinta[cite: 1]
- **วัตถุประสงค์:** เอกสารและสื่อการเรียนรู้สรุปทฤษฎีวิชา Machine Learning เพื่อการศึกษา[cite: 1]
