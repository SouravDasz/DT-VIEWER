from flask import Flask, Blueprint, render_template, request
from app.ml.Dt import tree

model_bp = Blueprint("model", __name__)

@model_bp.route("/iris", methods=["GET", "POST"])
def iris_model():

    if request.method == "GET":
        return render_template("iris.html", img_data=None)

    if request.method == "POST":
        criterion = request.form["criterion"]

        # Convert empty max_depth to None
        max_depth = request.form["max_depth"]
        max_depth = int(max_depth) if max_depth.strip() != "" else None

        min_samples_leaf = int(request.form["min_samples_leaf"])
        min_samples_split = int(request.form["min_samples_split"])

        acc, img_data, matric, depth, total_nodes = tree(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split
        )

        return render_template(
            "iris.html",
            acc=acc,
            img_data=img_data,    # ✅ base64 image sent
            matric=matric,
            depth=depth,
            total_nodes=total_nodes
        )
