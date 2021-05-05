
$(document).ready(function () {
  $('.animal-checkbox').each(function () {
    id = $(this).attr('id')
    imgId = "#" + id + "-img"

    if (this.checked) {
      if ($(imgId).hasClass('grayscale')) {
        $(imgId).removeClass('grayscale')
      }
    } else {
      if (!$(imgId).hasClass('grayscale')) {
        $(imgId).addClass('grayscale')
      }
    }
  });

  $(".animal-checkbox").change(function () {
    $('.animal-checkbox').each(function () {
      id = $(this).attr('id')
      imgId = "#" + id + "-img"

      if (this.checked) {
        if ($(imgId).hasClass('grayscale')) {
          $(imgId).removeClass('grayscale')
        }
      } else {
        if (!$(imgId).hasClass('grayscale')) {
          $(imgId).addClass('grayscale')
        }
      }
    })

  });
});