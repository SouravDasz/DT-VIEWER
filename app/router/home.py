from flask import Blueprint,request,render_template,redirect,jsonify

home_bp=Blueprint("home",__name__)

@home_bp.route("/")
def landing_page():
    return render_template("landing_page.html")

@home_bp.route("/home")
def home():
    return render_template("home.html")

