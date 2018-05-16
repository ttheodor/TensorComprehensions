
add_library(
  tc_lang
  STATIC

  parser.cc
  lexer.cc
  tc_format.cc
)

install(
  TARGETS
  tc_lang

  DESTINATION lib
)
