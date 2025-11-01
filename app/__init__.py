from flask import Flask, request, render_template

def create_app():
    app=Flask(__name__)

    from app.router.home import home_bp
    from app.router.model import model_bp
    app.register_blueprint(home_bp)
    app.register_blueprint(model_bp)


    return app