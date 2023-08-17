// add dropbox version to the docs
// A $( document ).ready() block.
$(document).ready(function () {
    // append right arrow to the version
    $(".version").append(" <i class='fa fa-angle-right'></i>");
    $(".version").append(" <i class='fa fa-angle-down' style='display: none;'></i>");
    var olive_version = {
        "latest": "https://microsoft.github.io/Olive",
        "0.2.1": "https://microsoft.github.io/Olive/0.2.1/",
        "0.2.0": "https://microsoft.github.io/Olive/0.2.0/",
        "0.1.0": "https://microsoft.github.io/Olive/0.1.0/",
    }
    for (var version in olive_version) {
        $(".version").append(
            "<div class='to-display-content' style='display: none;'><a class='v_link' href='"
            + olive_version[version] + "'>" + version + "</a></div>");
    }

    $(".fa").css({
        "cursor": "pointer",
        "color": "white"
    })

    $(".v_link").css({
        "margin-left": "-20px",
        "color": "white",
        "font-size": "14px",
    })

    // register click event: click on the version, show the version list
    $(".fa-angle-right").click(function () {
        $(".to-display-content").css("display", "block");
        $(".fa-angle-down").css("display", "inline");
        $(".fa-angle-right").css("display", "none");
    })

    // register click event: click on the version list, hide the version list
    $(".fa-angle-down").click(function () {
        $(".to-display-content").css("display", "none");
        $(".fa-angle-down").css("display", "none");
        $(".fa-angle-right").css("display", "inline");
    })
});
