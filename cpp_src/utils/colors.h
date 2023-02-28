#pragma once
#include <string>

namespace colors {
    // Regular text colors
    const std::string RESET = "\033[0m";
    const std::string BLACK = "\033[0;30m";
    const std::string RED = "\033[0;31m";
    const std::string GREEN = "\033[0;32m";
    const std::string YELLOW = "\033[0;33m";
    const std::string BLUE = "\033[0;34m";
    const std::string MAGENTA = "\033[0;35m";
    const std::string CYAN = "\033[0;36m";
    const std::string WHITE = "\033[0;37m";

    // Bold text colors
    const std::string BOLD_BLACK = "\033[1;30m";
    const std::string BOLD_RED = "\033[1;31m";
    const std::string BOLD_GREEN = "\033[1;32m";
    const std::string BOLD_YELLOW = "\033[1;33m";
    const std::string BOLD_BLUE = "\033[1;34m";
    const std::string BOLD_MAGENTA = "\033[1;35m";
    const std::string BOLD_CYAN = "\033[1;36m";
    const std::string BOLD_WHITE = "\033[1;37m";

    // Background colors
    const std::string BG_BLACK = "\033[40m";
    const std::string BG_RED = "\033[41m";
    const std::string BG_GREEN = "\033[42m";
    const std::string BG_YELLOW = "\033[43m";
    const std::string BG_BLUE = "\033[44m";
    const std::string BG_MAGENTA = "\033[45m";
    const std::string BG_CYAN = "\033[46m";
    const std::string BG_WHITE = "\033[47m";

    // Bold background colors
    const std::string BOLD_BG_BLACK = "\033[40;1m";
    const std::string BOLD_BG_RED = "\033[41;1m";
    const std::string BOLD_BG_GREEN = "\033[42;1m";
    const std::string BOLD_BG_YELLOW = "\033[43;1m";
    const std::string BOLD_BG_BLUE = "\033[44;1m";
    const std::string BOLD_BG_MAGENTA = "\033[45;1m";
    const std::string BOLD_BG_CYAN = "\033[46;1m";
    const std::string BOLD_BG_WHITE = "\033[47;1m";
}