function openImagePreview(imageSrc, title) {
  document.getElementById('previewImage').src = imageSrc;
  document.getElementById('previewTitle').textContent = title;
  document.getElementById('downloadBtn').href = imageSrc;
  var modal = new bootstrap.Modal(document.getElementById('imagePreviewModal'));
  modal.show();
}