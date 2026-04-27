from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_without_exceptions():
    app = AppTest.from_file("streamlit_app.py")

    app.run(timeout=20)

    assert len(app.exception) == 0
    assert any(title.value == "VibeFinder" for title in app.title)
    assert any(button.label == "Run Agent" for button in app.button)
