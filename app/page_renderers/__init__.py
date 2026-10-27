"""Pages package for GraphiSense."""

from app.page_renderers.modelling import render_modelling_page
from app.page_renderers.methodology import render_methodology_page, render_about_dysgraphia_page, render_repository_info_page, render_comparison_page
from app.page_renderers.preprocessing import render_preprocessing_page_wrapper
from app.page_renderers.repository_info import render_repository_info_page_wrapper
from app.page_renderers.about import render_about_page_wrapper
from app.page_renderers.comparison import render_comparison_page_wrapper

__all__ = [
    "render_modelling_page",
    "render_methodology_page",
    "render_about_dysgraphia_page",
    "render_repository_info_page",
    "render_comparison_page",
    "render_preprocessing_page_wrapper",
    "render_repository_info_page_wrapper",
    "render_about_page_wrapper",
    "render_comparison_page_wrapper",
]