// add dropbox version to the docs
// A $( document ).ready() block.
$(document).ready(function () {
    // Check if version element exists
    if ($(".version").length === 0) {
        console.log("Version element not found, version selector disabled");
        return;
    }

    // append right arrow to the version
    $(".version").append(" <i class='fa fa-angle-right'></i>");
    $(".version").append(" <i class='fa fa-angle-down' style='display: none;'></i>");
    // list the version under https://github.com/microsoft/Olive/tree/gh-pages
    var olive_version = {
        "latest": "https://microsoft.github.io/Olive",
    };
    
    // Function to finalize version selector setup
    function setupVersionSelector() {
        // sort the version list with reverse order
        var keys = Object.keys(olive_version);
        keys.sort(function (a, b) {
            return b.localeCompare(a);
        });

        for (var i = 0; i < keys.length; i++) {
            var version = keys[i];
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
    }

    $.ajax({
        url: "https://api.github.com/repos/microsoft/olive/contents?ref=gh-pages",
        type: "GET",
        dataType: "json",
        async: true,
        timeout: 5000,
        success: function (data) {
            for (var i = 0; i < data.length; i++) {
                var folder_name = data[i].name;
                // if folder_name is a version number, add it to the version list
                if (folder_name.match(/^\d+\.\d+\.\d+$/)) {
                    console.log(folder_name);
                    olive_version[folder_name] = "https://microsoft.github.io/Olive/" + folder_name + "/";
                }
            }
            setupVersionSelector();
        },
        error: function(xhr, status, error) {
            console.log("Failed to fetch versions from GitHub API:", error);
            console.log("Setting up version selector with only 'latest' version");
            setupVersionSelector();
        }
    });
});
