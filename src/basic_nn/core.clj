(ns basic-nn.core
  (:require [clojure.core.matrix :as matrix]
            [clojure.core.matrix.random :as random]
            [clojure.core.matrix.operators :refer :all]
            [clojure.math.numeric-tower :refer [expt abs]]
            ))

(matrix/set-current-implementation :vectorz)

(def training-input
  [[0 0 1]
   [1 1 1]
   [1 0 1]
   [0 1 1]])

(def training-output
  (matrix/transpose
   [[0 1 1 0]]))


(defn random-synapses
  "Create a matrix of random synaptic weights."
  [from to]
  (matrix/array
   (take from
         (repeatedly
          #(- (* 2 (take to (random/randoms))) 1)
          ))))


;; synapses are mutable
(def synapses-0 (atom (random-array 3 4)))
(def synapses-1 (atom (random-array 4 1)))

;; sigmoid fn
(defn sigmoid [x]
  (/ 1 (#(* % %) (- x))))

;; derivative of sigmoid
(defn deriv [x]
  (* x (- 1 x)))

(defn step-forward
  "For each step in feed forward, the new layer of neurons
  are a function of the previous layer and it's synapses"
  [neurons synapses]
  (sigmoid (matrix/dot neurons synapses)))

;; (step-forward
;;  (step-forward training-input synapses-0)
;;  synapses-1
;;  )

(defn feed-forward
  [input & synapses]
  (case (count synapses)
    0 input ; if no synapses return just the input
    1 (step-forward input @(first synapses))

    ;; otherwise recurse
    (apply
     feed-forward
     (into [(step-forward input @(first synapses))]
           (rest synapses)))))

(apply feed-forward
       (into
        [(apply feed-forward [training-input synapses-0])] (rest [ synapses-0 synapses-1])))

(defn errors [training-outputs real-outputs]
  (- training-outputs real-outputs))

(defn deltas [error neurons]
  (* (error training-outputs real-outputs)
     (deriv real-outputs)))

(defn apply-deltas [synapses neurons deltas]
  (+ @synapses (matrix/dot (matrix/transpose neurons) deltas)))

(defn output-deltas [targets outputs]
  (* (mapv deriv outputs)
     (- targets outputs)))

(defn hidden-deltas [output-deltas neurons synapses]
  (* (mapv deriv neurons)
     (mapv #(apply + %)
           (* output-deltas synapses))))

(defn update-synapses [deltas neurons synapses learning-rate]
  (+ synapses (* learning-rate
                 (mapv #(* deltas %) neurons))))

(defn train [training-input training-output]
  (let [output-layer (feed-forward training-input synapses-0 synapses-1)
        output-error (- training-output output-layer)
        output-delta (* output-error (deriv output-layer))
        hidden-layer (feed-forward training-input synapses-0)
        hidden-error (matrix/dot output-delta (matrix/transpose @synapses-1))
        hidden-delta (* hidden-error (deriv hidden-layer))]
    (do
      (reset! synapses-1
              (apply-deltas synapses-1 hidden-layer output-delta))
      (reset! synapses-0
              (apply-deltas synapses-0 training-input hidden-delta))
      (mean-error output-error))))


(defn mean-error [numbers]
  (let [absolutes (map abs (flatten numbers))]
    (/ (apply + absolutes) (count absolutes))))

(deref synapses-1)

;; Train once
(train training-input training-output)

;; synapses are mutable
(def synapses-0 (atom (random-array 3 4)))
(def synapses-1 (atom (random-array 4 1)))

;; Train 1000 passes
(doseq [i (range 1000)]
  (train training-input training-output))


(feed-forward training-input synapses-0 synapses-1)
(feed-forward (feed-forward training-input synapses-0) synapses-1)

(reset! synaptic-weights
        (calc-synaptic-weights (feed-forward) @synaptic-weights))
